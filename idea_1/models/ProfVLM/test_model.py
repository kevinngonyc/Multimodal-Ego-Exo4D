import unittest

import torch

from model import AttentiveGatedProjector, ProfVLM


class AttentiveGatedProjectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = AttentiveGatedProjector(
            input_dim=8, hidden_dim=12, out_dim=6, num_heads=2, dropout=0.0
        )

    def test_single_and_multiple_views(self):
        for views in (1, 4):
            with self.subTest(views=views):
                output = self.model(torch.randn(2, views, 8))
                self.assertEqual(output.shape, (2, 6))
                self.assertTrue(torch.isfinite(output).all())

    def test_gate_adds_scaled_normalized_features(self):
        # A zero gate logit gives sigmoid(0) = 0.5, so the residual
        # must send 1.5 times the normalized features to the next linear layer.
        torch.nn.init.zeros_(self.model.gate[0].weight)
        torch.nn.init.zeros_(self.model.gate[0].bias)
        captured = {}

        def capture_normalized(module, args, output):
            captured["normalized"] = output.detach().clone()

        def capture_projection_input(module, args):
            captured["projection_input"] = args[0].detach().clone()

        norm_hook = self.model.norm1.register_forward_hook(capture_normalized)
        proj_hook = self.model.proj2.register_forward_pre_hook(capture_projection_input)
        try:
            self.model(torch.randn(2, 3, 8))
        finally:
            norm_hook.remove()
            proj_hook.remove()
        torch.testing.assert_close(
            captured["projection_input"], 1.5 * captured["normalized"]
        )

    def test_view_permutation_invariance_and_batch_independence(self):
        self.model.eval()
        features = torch.randn(2, 4, 8)
        output = self.model(features)
        torch.testing.assert_close(output, self.model(features[:, [2, 0, 3, 1]]))
        separate = torch.cat([self.model(sample.unsqueeze(0)) for sample in features])
        torch.testing.assert_close(output, separate)

    def test_backward_reaches_inputs_and_all_parameters(self):
        features = torch.randn(2, 3, 8, requires_grad=True)
        self.model(features).square().mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_inputs(self):
        for shape in ((8,), (2, 8), (2, 7), (2, 3, 7), (2, 0, 8), (2, 3, 4, 8)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.model(torch.randn(*shape))

    def test_invalid_configuration(self):
        for kwargs in (
            {"input_dim": 7, "num_heads": 2},
            {"hidden_dim": 0},
            {"out_dim": -1},
            {"num_heads": 0},
            {"dropout": -0.1},
            {"dropout": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AttentiveGatedProjector(**kwargs)


class ProfVLMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.encoder = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Flatten(1),
            torch.nn.Linear(3, 400),
        )
        self.model = ProfVLM(self.encoder).eval()

    def test_views_match_individually_encoded_videos(self):
        for views in (1, 4):
            with self.subTest(views=views):
                # Transpose spatial dimensions to exercise non-contiguous inputs.
                videos = torch.randn(2, views, 3, 2, 4, 4).transpose(-1, -2)
                captured = {}

                def capture_features(module, args):
                    captured["features"] = args[0].detach().clone()

                hook = self.model.agp.register_forward_pre_hook(capture_features)
                try:
                    output = self.model(videos)
                finally:
                    hook.remove()
                expected_features = torch.stack(
                    [self.encoder(videos[:, view]) for view in range(views)], dim=1
                )
                torch.testing.assert_close(captured["features"], expected_features)
                self.assertEqual(output.shape, (2, 576))
                torch.testing.assert_close(output, self.model.agp(expected_features))

    def test_single_video_inputs_use_one_view(self):
        videos = torch.randn(2, 3, 2, 4, 4)
        torch.testing.assert_close(self.model(videos), self.model(videos.unsqueeze(1)))

    def test_backward_reaches_videos_and_encoder(self):
        videos = torch.randn(2, 4, 3, 2, 4, 4, requires_grad=True)
        self.model(videos).square().mean().backward()
        for gradient in (videos.grad, self.encoder[2].weight.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_invalid_video_shapes(self):
        for shape in ((2, 8), (2, 3, 4, 8), (2, 0, 3, 2, 4, 4)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.model(torch.randn(*shape))

    def test_invalid_encoder_output_shapes(self):
        for encoder in (torch.nn.Identity(), torch.nn.Flatten(1)):
            with self.subTest(encoder=encoder), self.assertRaisesRegex(
                ValueError, "Expected video_encoder output"
            ):
                ProfVLM(encoder)(torch.randn(2, 4, 3, 2, 4, 4))


if __name__ == "__main__":
    unittest.main()
