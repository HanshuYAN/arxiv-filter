# encoding: utf-8
import unittest

from rule_filter import DEFAULT_THRESHOLD, judge_paper, normalize, resolve_backend


class NormalizeTest(unittest.TestCase):
    def test_folds_case_hyphens_and_punctuation(self):
        self.assertEqual(normalize("Text-to-Video, Models!"), "text to video models")


class BackendTest(unittest.TestCase):
    def test_auto_uses_rules_without_cuda(self):
        self.assertEqual(resolve_backend("auto", cuda_available=False), "rules")

    def test_auto_uses_llm_with_cuda(self):
        self.assertEqual(resolve_backend("auto", cuda_available=True), "llm")

    def test_explicit_choice_is_kept(self):
        self.assertEqual(resolve_backend("rules", cuda_available=True), "rules")
        self.assertEqual(resolve_backend("llm", cuda_available=False), "llm")


class LexiconTest(unittest.TestCase):
    def assert_yes(self, title, abstract="", topic=None):
        judgment = judge_paper(title, abstract)
        self.assertEqual(judgment.answer, "YES", judgment.reason)
        if topic is not None:
            self.assertTrue(any(item.topic == topic for item in judgment.topics), judgment.reason)
        return judgment

    def assert_not(self, title, abstract=""):
        judgment = judge_paper(title, abstract)
        self.assertEqual(judgment.answer, "NOT", judgment.reason)
        return judgment

    def test_diffusion_model_in_the_title_is_yes(self):
        judgment = self.assert_yes(
            "Improving Virtual Try-On with Garment-focused Diffusion Models",
            topic="generative models",
        )
        self.assertGreaterEqual(judgment.score, 2 * 3)

    def test_strong_abstract_mention_is_enough(self):
        judgment = self.assert_yes("A study of sampling", "We train a diffusion model on images.")
        self.assertGreaterEqual(judgment.score, DEFAULT_THRESHOLD)

    def test_optical_flow_is_not_generative_flow(self):
        self.assert_not(
            "ScaleFlow++: Robust and Accurate Estimation of 3D Motion from Video",
            "We estimate optical flow and scene flow from video frames.",
        )

    def test_optical_flow_matching_does_not_count_as_flow_matching(self):
        self.assert_not("Optical Flow Matching", "We match optical flow fields across frames.")

    def test_flow_matching_is_yes(self):
        self.assert_yes("Flow Matching for Fast Generation", "Rectified flow matching trains generative models.")

    def test_diffusion_mri_is_not(self):
        self.assert_not(
            "Diffusion MRI tractography",
            "Diffusion weighted magnetic resonance imaging measures anisotropic diffusion.",
        )

    def test_variational_autoencoder_is_yes(self):
        self.assert_yes(
            "Disentangled Variational Autoencoder",
            "The VAE reconstructs materials.",
            topic="generative models",
        )

    def test_organ_does_not_match_gan(self):
        self.assert_not("Organ Segmentation", "We segment organs in CT scans.")

    def test_gan_is_yes(self):
        self.assert_yes("Image synthesis with GANs", "A generative adversarial network is trained.")

    def test_text_to_video_is_yes(self):
        judgment = self.assert_yes(
            "DiffTED: One-shot Audio-driven TED Talk Video Generation",
            "A text-to-video diffusion model drives gestures.",
            topic="video generation",
        )
        video = [item for item in judgment.topics if item.topic == "video generation"][0]
        phrases = {match["phrase"] for match in video.matches}
        self.assertTrue(
            phrases & {"text to video", "text to video generation", "video generation", "video diffusion"},
            phrases,
        )

    def test_overlapping_phrases_are_not_double_counted(self):
        judgment = judge_paper("Notes", "We study text-to-video generation.")
        self.assertEqual(judgment.answer, "YES")
        video = [item for item in judgment.topics if item.topic == "video generation"][0]
        self.assertEqual(video.score, 3)
        self.assertEqual(len(video.matches), 1)
        self.assertEqual(video.matches[0]["phrase"], "text to video generation")

    def test_repeated_phrase_counts_once_per_field(self):
        judgment = judge_paper("Notes", "Diffusion models improve diffusion models.")
        generative = [item for item in judgment.topics if item.topic == "generative models"][0]
        self.assertEqual(generative.score, 3)
        self.assertEqual(len(generative.matches), 1)

    def test_vision_language_is_yes(self):
        self.assert_yes(
            "ComAlign: Compositional Alignment in Vision-Language Models",
            topic="multimodal learning",
        )

    def test_robot_context_blocks_a_weak_language_model_match(self):
        # The same weak hits are YES without the robot guard (see the LLM RL test).
        # Blocking the abstract hit drops the score from 3 to 2.
        self.assert_not(
            "Reinforcement learning for language models",
            "Policy optimization aligns an LLM that controls a robot.",
        )

    def test_rlhf_and_preference_optimization_are_yes(self):
        self.assert_yes(
            "Alignment with Preference Optimization Is All You Need for LLM Safety",
            "We compare DPO and RLHF.",
            topic="post-training",
        )

    def test_llm_reinforcement_learning_is_yes(self):
        self.assert_yes(
            "Reinforcement learning for language models",
            "Policy optimization aligns an LLM with human feedback.",
        )

    def test_weak_diffusion_mention_needs_corroboration(self):
        abstract = "Diffusion sampling is analyzed in isolation."
        self.assert_not("A note", abstract)
        title = "Diffusion sampling schedules"
        self.assert_not(title, "No further detail.")
        both = judge_paper(title, abstract)
        self.assertEqual(both.answer, "YES", both.reason)

    def test_few_step_generation_is_yes(self):
        self.assert_yes("4-step diffusion sampling", "A consistency model distills the sampler.")

    def test_unrelated_paper_is_not(self):
        self.assert_not(
            "HSR-KAN: Efficient Hyperspectral Image Super-Resolution",
            "Kolmogorov-Arnold networks reconstruct hyperspectral images.",
        )

    def test_unknown_interest_uses_the_literal_phrase(self):
        interests = {"custom": ["neural radiance field"]}
        judgment = judge_paper(
            "Rendering",
            "A neural radiance field method for novel views.",
            interests=interests,
        )
        self.assertEqual(judgment.answer, "YES", judgment.reason)

    def test_threshold_can_be_lowered(self):
        judgment = judge_paper(
            "A note",
            "Diffusion sampling is analyzed in isolation.",
            threshold=1,
        )
        self.assertEqual(judgment.answer, "YES", judgment.reason)


if __name__ == "__main__":
    unittest.main()
