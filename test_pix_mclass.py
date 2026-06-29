"""Unit tests for pix_mclass — covers 2D and 3D segmentation paths.

Run with::

    python -m unittest test_pix_mclass

The test classes exercise:

* :class:`TestUNet2D` / :class:`TestUNet3D` — model construction and
  forward-pass shape correctness for 2D and 3D U-Net variants.
* :class:`TestGetModelDispatch` — the ``get_model`` factory dispatches to
  2D / 3D variants based on ``tridimensional_mode``.
* :class:`TestGetIterator` — the training iterator forwards
  ``tridimensional_mode`` to :class:`MultiChannelIterator` as
  ``n_spatial_dims=3``.
* :class:`TestAnisotropicDownscale` — encoder accepts a per-axis
  ``downscale`` tuple and the model still round-trips spatial shapes.
* :class:`TestInputShapeCap` — the ``input_shape`` argument of
  ``get_model`` caps per-axis downscale.
"""

from pathlib import Path
import os
import sys
import unittest

# Suppress GPU usage during tests
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Make sibling packages importable (mirrors the layout in ``test.py``)
_PATH_ROOT = Path(__file__).resolve().parents[1]
for _p in ("dataset_iterator", "pix_mclass"):
    _path = _PATH_ROOT.joinpath(_p)
    if _path.exists():
        sys.path.insert(0, str(_path))

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - tensorflow is a required runtime dep
    tf = None

from pix_mclass.unet import (
    get_model,
    get_unet,
    encoder_block,
    decoder_block,
    _is_downsampling,
    _normalize_input_shape,
    _resolve_level_downscale,
    _resolve_kernel_size,
)


@unittest.skipIf(tf is None, "tensorflow is required for U-Net tests")
class TestUNet2D(unittest.TestCase):
    """Verify 2D U-Net construction and forward pass."""

    n_classes = 3
    n_input_channels = 1
    spatial = (32, 32)

    def _build(self, **overrides):
        kwargs = dict(
            n_classes=self.n_classes,
            n_input_channels=self.n_input_channels,
            n_downsampling=2,
            filters=32,
            filters_min=16,
            tridimensional_mode=False,
        )
        kwargs.update(overrides)
        return get_model("unet", **kwargs)

    def test_input_rank_is_4(self):
        model = self._build()
        self.assertEqual(len(model.input.shape), 4, "2D model expects (B, H, W, C)")
        self.assertEqual(model.input.shape[-1], self.n_input_channels)

    def test_output_rank_and_classes(self):
        model = self._build()
        self.assertEqual(len(model.output.shape), 4)
        self.assertEqual(model.output.shape[-1], self.n_classes)

    def test_no_3d_layers(self):
        model = self._build()
        for layer in model.layers:
            self.assertNotIsInstance(layer, tf.keras.layers.Conv3D)
            self.assertNotIsInstance(layer, tf.keras.layers.Conv3DTranspose)
            self.assertNotIsInstance(layer, tf.keras.layers.MaxPool3D)

    def test_forward_pass_shape(self):
        model = self._build()
        x = np.random.rand(2, *self.spatial, self.n_input_channels).astype(np.float32)
        y = model(x, training=False).numpy()
        self.assertEqual(y.shape, (2, *self.spatial, self.n_classes))
        self.assertTrue(np.all(y >= 0))
        np.testing.assert_allclose(y.sum(axis=-1), 1.0, atol=1e-4)


@unittest.skipIf(tf is None, "tensorflow is required for U-Net tests")
class TestUNet3D(unittest.TestCase):
    """Verify 3D U-Net construction and forward pass."""

    n_classes = 2
    n_input_channels = 1
    spatial = (16, 16, 16)

    def _build(self, **overrides):
        kwargs = dict(
            n_classes=self.n_classes,
            n_input_channels=self.n_input_channels,
            n_downsampling=2,
            filters=16,
            filters_min=8,
            tridimensional_mode=True,
            input_shape=self.spatial,  # 3D defaults to WGN -> needs a known feature-level size
        )
        kwargs.update(overrides)
        return get_model("unet", **kwargs)

    def test_input_rank_is_5(self):
        model = self._build()
        self.assertEqual(len(model.input.shape), 5, "3D model expects (B, D, H, W, C)")
        self.assertEqual(model.input.shape[-1], self.n_input_channels)

    def test_output_rank_and_classes(self):
        model = self._build()
        self.assertEqual(len(model.output.shape), 5)
        self.assertEqual(model.output.shape[-1], self.n_classes)

    def test_no_2d_conv_layers(self):
        model = self._build()
        for layer in model.layers:
            self.assertNotIsInstance(layer, tf.keras.layers.Conv2D)
            self.assertNotIsInstance(layer, tf.keras.layers.Conv2DTranspose)
            self.assertNotIsInstance(layer, tf.keras.layers.MaxPool2D)

    def test_uses_3d_layers(self):
        model = self._build()
        has_conv3d = any(isinstance(l, tf.keras.layers.Conv3D) for l in model.layers)
        has_conv3d_t = any(isinstance(l, tf.keras.layers.Conv3DTranspose) for l in model.layers)
        self.assertTrue(has_conv3d, "3D model should contain Conv3D layers")
        self.assertTrue(has_conv3d_t, "3D model should contain Conv3DTranspose layers")

    def test_forward_pass_shape(self):
        model = self._build()
        x = np.random.rand(1, *self.spatial, self.n_input_channels).astype(np.float32)
        y = model(x, training=False).numpy()
        self.assertEqual(y.shape, (1, *self.spatial, self.n_classes))
        np.testing.assert_allclose(y.sum(axis=-1), 1.0, atol=1e-4)

    def test_maxpool_variant(self):
        model = self._build(maxpool=True)
        has_mp = any(isinstance(l, tf.keras.layers.MaxPool3D) for l in model.layers)
        self.assertTrue(has_mp, "maxpool=True must yield MaxPool3D layers in 3D mode")


@unittest.skipIf(tf is None, "tensorflow is required for U-Net tests")
class TestGetModelDispatch(unittest.TestCase):
    """Lower-level dispatch checks on the layer-builder helpers."""

    def test_unknown_architecture_raises(self):
        with self.assertRaises(ValueError):
            get_model("not_an_arch", n_classes=2, tridimensional_mode=False)

    def test_get_unet_2d_default(self):
        model = get_unet(
            n_classes=2,
            encoder_settings=[[{"filters": 8}, {"filters": 8, "downscale": 2, "maxpool": False}]],
            feature_settings=[{"filters": 8}],
            decoder_settings=[{"filters": 8}],
            tridimensional_mode=False,
        )
        self.assertEqual(len(model.input.shape), 4)

    def test_get_unet_3d(self):
        model = get_unet(
            n_classes=2,
            encoder_settings=[[{"filters": 8}, {"filters": 8, "downscale": 2, "maxpool": False}]],
            feature_settings=[{"filters": 8}],
            decoder_settings=[{"filters": 8}],
            tridimensional_mode=True,
        )
        self.assertEqual(len(model.input.shape), 5)

    def test_wgn_without_input_shape_raises(self):
        # WGN derives its window from the feature-level size; without input_shape it must error.
        with self.assertRaises(ValueError):
            get_model("unet", n_classes=2, n_downsampling=2, filters=16, filters_min=8,
                      tridimensional_mode=True, normalization="wgn")
        # but providing input_shape (or an explicit window) is fine
        get_model("unet", n_classes=2, n_downsampling=2, filters=16, filters_min=8,
                  tridimensional_mode=True, normalization="wgn", input_shape=(16, 16, 16))
        get_model("unet", n_classes=2, n_downsampling=2, filters=16, filters_min=8,
                  tridimensional_mode=True, normalization="wgn", norm_window_size=(1, 4, 4))

    def test_multi_input_3d(self):
        model = get_model(
            "unet",
            n_classes=2,
            n_inputs=2,
            n_input_channels=1,
            n_downsampling=2,
            filters=16,
            filters_min=8,
            tridimensional_mode=True,
            input_shape=(16, 16, 16),  # 3D defaults to WGN -> needs a known feature-level size
        )
        self.assertEqual(len(model.inputs), 2)
        for inp in model.inputs:
            self.assertEqual(len(inp.shape), 5)


# ---------------------------------------------------------------------------
# Anisotropic downscale (per-axis tuple) tests
# ---------------------------------------------------------------------------


@unittest.skipIf(tf is None, "tensorflow is required for U-Net tests")
class TestAnisotropicDownscale(unittest.TestCase):
    """Encoder/decoder accept tuple ``downscale`` differing across axes."""

    @staticmethod
    def _exact_type_layers(model, layer_cls):
        # Conv2DTranspose subclasses Conv2D in TF — use exact type checks
        return [l for l in model.layers if type(l) is layer_cls]

    def test_2d_anisotropic_strides(self):
        # downscale 1 on axis 0, 2 on axis 1
        encoder_settings = [
            [
                {"filters": 8},
                {"filters": 8, "downscale": (1, 2), "maxpool": False},
            ]
        ]
        model = get_unet(
            n_classes=2,
            encoder_settings=encoder_settings,
            feature_settings=[{"filters": 8}],
            decoder_settings=[{"filters": 8}],
        )
        # The single strided Conv2D should have strides (1, 2)
        strided = [
            l for l in self._exact_type_layers(model, tf.keras.layers.Conv2D)
            if l.strides != (1, 1)
        ]
        self.assertEqual(len(strided), 1)
        self.assertEqual(strided[0].strides, (1, 2))
        # Decoder Conv2DTranspose mirrors with strides (1, 2)
        upsamplers = self._exact_type_layers(model, tf.keras.layers.Conv2DTranspose)
        self.assertEqual(len(upsamplers), 1)
        self.assertEqual(upsamplers[0].strides, (1, 2))

    def test_3d_anisotropic_strides(self):
        # Common biomedical case: don't downsample Z, only XY
        encoder_settings = [
            [
                {"filters": 8},
                {"filters": 8, "downscale": (1, 2, 2), "maxpool": False},
            ]
        ]
        model = get_unet(
            n_classes=2,
            encoder_settings=encoder_settings,
            feature_settings=[{"filters": 8}],
            decoder_settings=[{"filters": 8}],
            tridimensional_mode=True,
        )
        strided = [
            l for l in self._exact_type_layers(model, tf.keras.layers.Conv3D)
            if l.strides != (1, 1, 1)
        ]
        self.assertEqual(len(strided), 1)
        self.assertEqual(strided[0].strides, (1, 2, 2))
        upsamplers = self._exact_type_layers(model, tf.keras.layers.Conv3DTranspose)
        self.assertEqual(len(upsamplers), 1)
        self.assertEqual(upsamplers[0].strides, (1, 2, 2))

    def test_3d_anisotropic_forward_pass(self):
        encoder_settings = [
            [{"filters": 4}, {"filters": 4, "downscale": (1, 2, 2), "maxpool": False}],
            [{"filters": 8}, {"filters": 8, "downscale": (1, 2, 2), "maxpool": False}],
        ]
        model = get_unet(
            n_classes=2,
            encoder_settings=encoder_settings,
            feature_settings=[{"filters": 8}],
            decoder_settings=[{"filters": 4}, {"filters": 8}],
            tridimensional_mode=True,
        )
        # Z axis is preserved end-to-end; XY is downsampled by 4 then upsampled
        x = np.random.rand(1, 3, 16, 16, 1).astype(np.float32)
        y = model(x, training=False).numpy()
        self.assertEqual(y.shape, (1, 3, 16, 16, 2))

    def test_2d_anisotropic_with_maxpool(self):
        encoder_settings = [
            [
                {"filters": 8},
                {"filters": 8, "downscale": (2, 1), "maxpool": True},
            ]
        ]
        model = get_unet(
            n_classes=2,
            encoder_settings=encoder_settings,
            feature_settings=[{"filters": 8}],
            decoder_settings=[{"filters": 8}],
        )
        mps = self._exact_type_layers(model, tf.keras.layers.MaxPool2D)
        self.assertEqual(len(mps), 1)
        self.assertEqual(mps[0].pool_size, (2, 1))

    def test_is_downsampling_helper(self):
        self.assertFalse(_is_downsampling(1))
        self.assertTrue(_is_downsampling(2))
        self.assertFalse(_is_downsampling((1, 1)))
        self.assertFalse(_is_downsampling((1, 1, 1)))
        self.assertTrue(_is_downsampling((1, 2)))
        self.assertTrue(_is_downsampling((1, 2, 1)))


# ---------------------------------------------------------------------------
# input_shape -> downscale capping tests
# ---------------------------------------------------------------------------


@unittest.skipIf(tf is None, "tensorflow is required for U-Net tests")
class TestInputShapeCap(unittest.TestCase):
    """``input_shape`` caps per-axis downscale in encoder levels."""

    # ---- pure-logic tests on the helpers (no model build) -------------------

    def test_normalize_input_shape_none(self):
        self.assertIsNone(_normalize_input_shape(None, 2))

    def test_normalize_input_shape_int_broadcast(self):
        self.assertEqual(_normalize_input_shape(64, 3), [64, 64, 64])

    def test_normalize_input_shape_zero_is_unconstrained(self):
        self.assertEqual(_normalize_input_shape((0, 64, 64), 3), [None, 64, 64])

    def test_normalize_input_shape_none_in_tuple(self):
        self.assertEqual(_normalize_input_shape((None, 64, 64), 3), [None, 64, 64])

    def test_normalize_input_shape_wrong_length_raises(self):
        with self.assertRaises(AssertionError):
            _normalize_input_shape((64, 64), 3)

    def test_resolve_level_downscale_unconstrained_returns_int(self):
        shape = None
        ds = _resolve_level_downscale(shape, 3, default=2)
        self.assertEqual(ds, 2)

    def test_resolve_level_downscale_all_above_threshold_returns_int(self):
        shape = [64, 64, 64]
        ds = _resolve_level_downscale(shape, 3, default=2)
        self.assertEqual(ds, 2)
        self.assertEqual(shape, [32, 32, 32])

    def test_resolve_level_downscale_caps_small_axis(self):
        shape = [1, 64, 64]
        ds = _resolve_level_downscale(shape, 3, default=2)
        self.assertEqual(ds, (1, 2, 2))
        self.assertEqual(shape, [1, 32, 32])

    def test_resolve_level_downscale_unconstrained_axis_passthrough(self):
        shape = [None, 32, 32]
        ds = _resolve_level_downscale(shape, 3, default=2)
        self.assertEqual(ds, 2)  # all axes get default since None counts as available
        # only constrained axes are floor-divided
        self.assertEqual(shape, [None, 16, 16])

    def test_resolve_level_downscale_progressive_cap(self):
        # Simulates 4 levels for input_shape (8, 256, 256) -> last level caps Z
        shape = [8, 256, 256]
        levels = [_resolve_level_downscale(shape, 3, default=2) for _ in range(4)]
        self.assertEqual(levels[0], 2)
        self.assertEqual(levels[1], 2)
        self.assertEqual(levels[2], 2)
        # at level 3 axis 0 has size 1, can no longer downsample
        self.assertEqual(levels[3], (1, 2, 2))
        self.assertEqual(shape, [1, 16, 16])

    # ---- end-to-end build tests --------------------------------------------

    @staticmethod
    def _strided_conv_layers(model, conv_cls):
        # Conv2DTranspose subclasses Conv2D — use exact type to count encoder convs only
        unit = (1,) * (3 if conv_cls is tf.keras.layers.Conv3D else 2)
        return [
            l for l in model.layers
            if type(l) is conv_cls and getattr(l, "strides", unit) != unit
        ]

    def test_input_shape_caps_downscale_3d(self):
        # n_downsampling=4 with Z=2 should cap Z after first level
        model = get_model(
            "unet",
            n_classes=2,
            n_input_channels=1,
            n_downsampling=4,
            filters=16,
            filters_min=8,
            tridimensional_mode=True,
            input_shape=(2, 64, 64),
        )
        strides = [l.strides for l in self._strided_conv_layers(model, tf.keras.layers.Conv3D)]
        self.assertEqual(len(strides), 4)
        # Level 0: Z=2 can downscale once -> (2,2,2). Then Z=1 -> (1,2,2) for the rest.
        self.assertEqual(strides[0], (2, 2, 2))
        for s in strides[1:]:
            self.assertEqual(s, (1, 2, 2))

    def test_input_shape_zero_axis_unconstrained_3d(self):
        # Z marked unconstrained -> all levels use isotropic downscale
        model = get_model(
            "unet",
            n_classes=2,
            n_input_channels=1,
            n_downsampling=3,
            filters=16,
            filters_min=8,
            tridimensional_mode=True,
            input_shape=(0, 64, 64),
            normalization="batch_norm",  # unconstrained Z: WGN window not derivable, irrelevant here
        )
        strides = [l.strides for l in self._strided_conv_layers(model, tf.keras.layers.Conv3D)]
        self.assertEqual(len(strides), 3)
        for s in strides:
            self.assertEqual(s, (2, 2, 2))

    def test_input_shape_caps_downscale_2d(self):
        # Tiny H axis -> first level downsamples both, then H is capped
        model = get_model(
            "unet",
            n_classes=2,
            n_input_channels=1,
            n_downsampling=3,
            filters=16,
            filters_min=8,
            input_shape=(2, 32),
        )
        strides = [l.strides for l in self._strided_conv_layers(model, tf.keras.layers.Conv2D)]
        self.assertEqual(len(strides), 3)
        self.assertEqual(strides[0], (2, 2))  # 2,32 -> 1,16
        self.assertEqual(strides[1], (1, 2))  # 1,16 -> 1,8
        self.assertEqual(strides[2], (1, 2))  # 1,8  -> 1,4

    def test_input_shape_does_not_change_model_input(self):
        # Only downscale should be affected; the keras Input remains variable.
        model = get_model(
            "unet",
            n_classes=2,
            n_input_channels=1,
            n_downsampling=2,
            filters=16,
            filters_min=8,
            tridimensional_mode=True,
            input_shape=(8, 64, 64),
        )
        # All spatial input dims remain None (variable)
        self.assertEqual(tuple(model.input.shape), (None, None, None, None, 1))

    def test_input_shape_with_anisotropy_forward_pass(self):
        # End-to-end: model built for input_shape (4, 32, 32) accepts that input
        model = get_model(
            "unet",
            n_classes=2,
            n_input_channels=1,
            n_downsampling=3,
            filters=16,
            filters_min=8,
            tridimensional_mode=True,
            input_shape=(4, 32, 32),
        )
        x = np.random.rand(1, 4, 32, 32, 1).astype(np.float32)
        y = model(x, training=False).numpy()
        self.assertEqual(y.shape, (1, 4, 32, 32, 2))

    def test_input_shape_wrong_length_raises(self):
        with self.assertRaises(AssertionError):
            get_model(
                "unet",
                n_classes=2,
                n_input_channels=1,
                n_downsampling=2,
                filters=16,
                filters_min=8,
                tridimensional_mode=True,
                input_shape=(64, 64),  # only 2 entries for 3D mode
            )


# ---------------------------------------------------------------------------
# Kernel-size capping tests
# ---------------------------------------------------------------------------


@unittest.skipIf(tf is None, "tensorflow is required for U-Net tests")
class TestKernelSizeCap(unittest.TestCase):
    """``input_shape`` caps per-axis conv kernel size (odd, lower-better)."""

    # ---- pure-logic helper tests -----------------------------------------

    def test_unconstrained_returns_target(self):
        self.assertEqual(_resolve_kernel_size(5, None, 2), 5)

    def test_size_4_caps_to_3(self):
        # User-specified case: dim=4 -> max kernel is 3
        self.assertEqual(_resolve_kernel_size(5, [4, 4], 2), 3)
        self.assertEqual(_resolve_kernel_size(3, [4, 4], 2), 3)

    def test_size_8_keeps_5(self):
        self.assertEqual(_resolve_kernel_size(5, [8, 8], 2), 5)

    def test_small_axis_drops_to_1(self):
        # dim=2, ker=3: 2 < 2*(3-1)=4, drop -> 1
        self.assertEqual(_resolve_kernel_size(3, [2, 2], 2), 1)

    def test_size_1_drops_to_1(self):
        self.assertEqual(_resolve_kernel_size(5, [1, 64], 2), (1, 5))
        self.assertEqual(_resolve_kernel_size(3, [1, 64], 2), (1, 3))

    def test_even_target_forced_to_odd(self):
        # ker=4 -> 3 first; then check fit (no cap on dim 64)
        self.assertEqual(_resolve_kernel_size(4, [64, 64], 2), 3)
        # ker=4 with tiny dim collapses to 1
        self.assertEqual(_resolve_kernel_size(4, [2, 2], 2), 1)

    def test_anisotropic_returns_tuple(self):
        # Z=2 -> 1; XY=64 -> 3
        self.assertEqual(_resolve_kernel_size(3, [2, 64, 64], 3), (1, 3, 3))
        # Z=4 -> 3; XY=64 -> 5
        self.assertEqual(_resolve_kernel_size(5, [4, 64, 64], 3), (3, 5, 5))

    def test_unconstrained_axis_keeps_target(self):
        # None on axis 0, constrained on other axes
        self.assertEqual(_resolve_kernel_size(5, [None, 4, 4], 3), (5, 3, 3))

    def test_per_axis_target(self):
        # Tuple target should be honored axis-by-axis
        self.assertEqual(_resolve_kernel_size((3, 5, 5), [4, 4, 4], 3), 3)

    def test_per_axis_target_wrong_length_raises(self):
        with self.assertRaises(AssertionError):
            _resolve_kernel_size((3, 5), [4, 4, 4], 3)

    def test_progressive_reduction_7_to_5_to_3(self):
        # dim=12, ker=7: 12 >= 2*6=12 OK -> stays 7
        self.assertEqual(_resolve_kernel_size(7, [12, 12], 2), 7)
        # dim=10, ker=7: 10 < 12 -> 5; 10 >= 8 OK -> 5
        self.assertEqual(_resolve_kernel_size(7, [10, 10], 2), 5)
        # dim=6, ker=7: 6 < 12 -> 5; 6 < 8 -> 3; 6 >= 4 OK -> 3
        self.assertEqual(_resolve_kernel_size(7, [6, 6], 2), 3)

    # ---- end-to-end model build tests ------------------------------------

    @staticmethod
    def _conv_kernels(model, conv_cls):
        # Conv2DTranspose subclasses Conv2D — use exact type
        return [
            (l.name, l.kernel_size)
            for l in model.layers
            if type(l) is conv_cls
        ]

    def test_3d_input_shape_caps_kernels(self):
        # Z=4 -> ker capped to 3 on Z; XY large -> ker stays as default
        model = get_model(
            "unet",
            n_classes=2,
            n_input_channels=1,
            n_downsampling=2,
            filters=16,
            filters_min=8,
            tridimensional_mode=True,
            input_shape=(4, 64, 64),
        )
        kernels = self._conv_kernels(model, tf.keras.layers.Conv3D)
        # All Conv3D in encoder/decoder/output should have ker[0] <= 3 on Z
        self.assertTrue(len(kernels) > 0)
        for name, ks in kernels:
            self.assertLessEqual(ks[0], 3, msg=f"{name} has ker[0]={ks[0]} on Z=4")
            # XY axes can still be 3 or 5
            self.assertIn(ks[1], (1, 3, 5), msg=f"{name} bad ker[1]={ks[1]}")

    def test_3d_tiny_z_forces_kernel_1(self):
        # Z=2 forces kernel along Z to 1 everywhere
        model = get_model(
            "unet",
            n_classes=2,
            n_input_channels=1,
            n_downsampling=2,
            filters=16,
            filters_min=8,
            tridimensional_mode=True,
            input_shape=(2, 64, 64),
        )
        kernels = self._conv_kernels(model, tf.keras.layers.Conv3D)
        for name, ks in kernels:
            self.assertEqual(ks[0], 1, msg=f"{name} ker[0]={ks[0]} on Z=2 must be 1")

    def test_kernels_are_odd_in_2d(self):
        model = get_model(
            "unet",
            n_classes=2,
            n_input_channels=1,
            n_downsampling=3,
            filters=16,
            filters_min=8,
            input_shape=(8, 64),
        )
        kernels = self._conv_kernels(model, tf.keras.layers.Conv2D)
        for name, ks in kernels:
            for k in ks:
                self.assertEqual(k % 2, 1, msg=f"{name} kernel must be odd, got {ks}")

    def test_unconstrained_input_keeps_default_kernels(self):
        model = get_model(
            "unet",
            n_classes=2,
            n_input_channels=1,
            n_downsampling=2,
            filters=16,
            filters_min=8,
            tridimensional_mode=True,
            normalization="batch_norm",  # unconstrained input: avoid WGN's feature-size-derived window
        )
        kernels = self._conv_kernels(model, tf.keras.layers.Conv3D)
        seen = {ks for _, ks in kernels}
        # No tuples should appear when input_shape is unconstrained
        for ks in seen:
            self.assertEqual(ks[0], ks[1])
            self.assertEqual(ks[1], ks[2])
        # The encoder still uses 3 and 5 plus the output_conv at 3
        scalar_kernels = {ks[0] for ks in seen}
        self.assertTrue(scalar_kernels.issubset({3, 5}))

    def test_forward_pass_with_kernel_capping(self):
        # End-to-end with anisotropic Z and capped kernels
        model = get_model(
            "unet",
            n_classes=2,
            n_input_channels=1,
            n_downsampling=3,
            filters=16,
            filters_min=8,
            tridimensional_mode=True,
            input_shape=(4, 32, 32),
        )
        x = np.random.rand(1, 4, 32, 32, 1).astype(np.float32)
        y = model(x, training=False).numpy()
        self.assertEqual(y.shape, (1, 4, 32, 32, 2))

    def test_output_kernel_size_propagated_to_get_unet(self):
        # Direct call: output_kernel_size kwarg controls the final conv
        model = get_unet(
            n_classes=2,
            encoder_settings=[[{"filters": 8}, {"filters": 8, "downscale": 2, "maxpool": False}]],
            feature_settings=[{"filters": 8}],
            decoder_settings=[{"filters": 8}],
            output_kernel_size=(1, 3),
        )
        out = next(l for l in model.layers if l.name == "output_conv")
        self.assertEqual(out.kernel_size, (1, 3))


# ---------------------------------------------------------------------------
# Iterator tests
#
# We exercise ``get_iterator`` with a minimal in-memory ``DatasetIO`` so the
# test does not depend on an .h5 file. The goal is to verify that
# ``tridimensional_mode`` is forwarded to :class:`MultiChannelIterator` as
# ``n_spatial_dims=3``.
# ---------------------------------------------------------------------------

try:
    from dataset_iterator.datasetIO.datasetIO import DatasetIO
    from pix_mclass.training import get_iterator
    _ITERATOR_DEPS_OK = True
except Exception:  # pragma: no cover - exercised only if deps are missing
    DatasetIO = object  # type: ignore[assignment,misc]
    _ITERATOR_DEPS_OK = False


class _InMemoryDataset:
    """Minimal stand-in for an h5py dataset (supports ``shape`` and slicing)."""

    def __init__(self, array):
        self._array = np.asarray(array)

    @property
    def shape(self):
        return self._array.shape

    def __len__(self):
        return self._array.shape[0]

    def __getitem__(self, item):
        return np.copy(self._array[item])


class _InMemoryDatasetIO(DatasetIO):  # type: ignore[misc]
    """Minimal :class:`DatasetIO` returning numpy arrays directly from a dict."""

    def __init__(self, datasets):
        super().__init__()
        self._datasets = {k: _InMemoryDataset(v) for k, v in datasets.items()}

    def close(self):
        return None

    def get_dataset_paths(self, channel_keyword, group_keyword):
        return [
            p for p in self._datasets
            if (channel_keyword is None or p.endswith(channel_keyword))
            and (group_keyword is None or group_keyword in p)
        ]

    def get_dataset(self, path):
        return self._datasets[path]

    def get_attribute(self, path, attribute_name):
        return None

    def __contains__(self, key):
        return key in self._datasets

    def get_parent_path(self, path):
        return path.rsplit("/", 1)[0] if "/" in path else ""


@unittest.skipIf(not _ITERATOR_DEPS_OK, "dataset_iterator / pix_mclass.training import failed")
class TestGetIterator(unittest.TestCase):
    """Verify ``tridimensional_mode`` is forwarded to the iterator."""

    @staticmethod
    def _make_2d_io(n=4, h=16, w=16, n_classes=3):
        rng = np.random.default_rng(0)
        raw = rng.random(size=(n, h, w), dtype=np.float32)
        classes = rng.integers(0, n_classes + 1, size=(n, h, w)).astype(np.float32)
        return _InMemoryDatasetIO({"/group/raw": raw, "/group/classes": classes})

    @staticmethod
    def _make_3d_io(n=2, d=8, h=16, w=16, n_classes=3):
        rng = np.random.default_rng(1)
        raw = rng.random(size=(n, d, h, w), dtype=np.float32)
        classes = rng.integers(0, n_classes + 1, size=(n, d, h, w)).astype(np.float32)
        return _InMemoryDatasetIO({"/group/raw": raw, "/group/classes": classes})

    def test_default_is_2d(self):
        it = get_iterator(self._make_2d_io(), scaling_data_generator=None, batch_size=2)
        self.assertEqual(it.n_spatial_dims, 2)

    def test_tridimensional_mode_sets_n_spatial_dims_to_3(self):
        it = get_iterator(
            self._make_3d_io(),
            scaling_data_generator=None,
            batch_size=1,
            tridimensional_mode=True,
        )
        self.assertEqual(it.n_spatial_dims, 3)

    def test_2d_iterator_yields_batches(self):
        it = get_iterator(self._make_2d_io(), scaling_data_generator=None, batch_size=2)
        x, y = it[0]
        # x: (B, H, W, C_in); y: (B, H, W, 2)  -- output channel is concatenated with mask
        self.assertEqual(x.ndim, 4)
        self.assertEqual(y.ndim, 4)
        self.assertEqual(x.shape[0], 2)

    def test_3d_iterator_yields_batches(self):
        it = get_iterator(
            self._make_3d_io(),
            scaling_data_generator=None,
            batch_size=1,
            tridimensional_mode=True,
        )
        x, y = it[0]
        self.assertEqual(x.ndim, 5, "3D batch must have rank 5: (B, D, H, W, C)")
        self.assertEqual(y.ndim, 5)
        self.assertEqual(x.shape[0], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
