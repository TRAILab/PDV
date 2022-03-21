#!/usr/bin/env python

import unittest

import torch
from easydict import EasyDict

from pcdet.models.model_utils.attention_utils import (TransformerEncoder,
                                                      FrequencyPositionalEncoding3d,
                                                      FeedForwardPositionalEncoding)


class TestTransformerEncoder(unittest.TestCase):
    def src_key_padding_mask_eval(self, input, encoder, input_point_locations=None):
        input_copy = input.clone()
        output = encoder(input, positional_input=input_point_locations, src_key_padding_mask=None)

        # Check that zero padding mask does not change output
        src_key_padding_mask_false = torch.zeros(input.shape[0], input.shape[1], dtype=torch.bool)
        output_masked_false = encoder(input, positional_input=input_point_locations, src_key_padding_mask=src_key_padding_mask_false)
        # Check input is unchanged
        self.assertTrue(torch.all(torch.eq(input, input_copy)))
        self.assertTrue(torch.all(torch.eq(output, output_masked_false)))

        # Check that true padding mask masks all output in a batch
        src_key_padding_mask_true = torch.ones(input.shape[0], input.shape[1], dtype=torch.bool)
        src_key_padding_mask_true[0:2] = False
        output_masked_true = encoder(input, positional_input=input_point_locations, src_key_padding_mask=src_key_padding_mask_true)
        # Check input is unchanged
        self.assertTrue(torch.all(torch.eq(input, input_copy)))
        # Unmasked output should be exactly the same the regular output
        self.assertTrue(torch.allclose(output[0:2], output_masked_true[0:2], atol=1e-7))
        # Masked output should be untouched
        self.assertTrue(torch.all(torch.eq(input[2:], output_masked_true[2:])))

        # Check that masked inputs are untouched
        input_default = torch.randn_like(input)
        random_vals = torch.randn(1, input_default.shape[-1])
        random_vals = random_vals.repeat(input_default.shape[0], 1)
        # Change index 3, 5 to same random input to check the same input results in the same output
        input_default[:, 3] = random_vals
        input_default[:, 5] = random_vals
        input_default_copy = input_default.clone()
        src_key_padding_mask_default = torch.zeros(input.shape[0], input.shape[1], dtype=torch.bool)
        src_key_padding_mask_default[:, 3] = True
        src_key_padding_mask_default[:, 5] = True
        output_masked_default = encoder(input_default, positional_input=input_point_locations, src_key_padding_mask=src_key_padding_mask_default)
        # Check input is unchanged
        self.assertTrue(torch.all(torch.eq(input_default, input_default_copy)))
        # Check that the untouched outputs result in the same values after shared FFN in transfomer encoder
        self.assertTrue(torch.allclose(output_masked_default[:, 3], output_masked_default[:, 5]))

    def test_transformer_encoder_default(self):
        attention_cfg = EasyDict(dict(NUM_FEATURES=6, NUM_HEADS=1, NUM_HIDDEN_FEATURES=16, NUM_LAYERS=1, DROPOUT=0.))
        d_input = 3
        input = torch.randn(10, 27, attention_cfg.NUM_FEATURES)

        # Default encoder
        encoder = TransformerEncoder(attention_cfg)
        # Check if output is the exact same shape as input
        output = encoder(input, positional_input=None)
        self.assertTrue(torch.all(torch.eq(torch.tensor(input.shape), torch.tensor(output.shape))))

        # Encoder with frequency positional encoding
        fpe = FrequencyPositionalEncoding3d(attention_cfg.NUM_FEATURES, max_spatial_shape=torch.IntTensor([3, 3, 3]), dropout=0.)
        encoder_fpe = TransformerEncoder(attention_cfg, fpe)
        # Check if output is the exact same shape as input
        output = encoder_fpe(input, positional_input=None)
        self.assertTrue(torch.all(torch.eq(torch.tensor(input.shape), torch.tensor(output.shape))))

        # Encoder with feed forward positional encoding
        ffn = FeedForwardPositionalEncoding(d_input=d_input, d_output=attention_cfg.NUM_FEATURES)
        encoder_ffn = TransformerEncoder(attention_cfg, ffn)
        # Check if output is the exact same shape as input
        positional_input = torch.randn(10, 27, d_input)
        output = encoder_ffn(input, positional_input)
        self.assertTrue(torch.all(torch.eq(torch.tensor(input.shape), torch.tensor(output.shape))))

    def test_src_key_padding_mask_no_positional_encoding(self):
        attention_cfg = EasyDict(dict(NUM_FEATURES=6, NUM_HEADS=1, NUM_HIDDEN_FEATURES=16, NUM_LAYERS=1, DROPOUT=0.))
        input = torch.randn(10, 27, attention_cfg.NUM_FEATURES)
        encoder = TransformerEncoder(attention_cfg)
        self.src_key_padding_mask_eval(input, encoder)

    def test_src_key_padding_mask_frequency(self):
        attention_cfg = EasyDict(dict(NUM_FEATURES=6, NUM_HEADS=1, NUM_HIDDEN_FEATURES=16, NUM_LAYERS=1, DROPOUT=0.))
        input = torch.randn(10, 8, attention_cfg.NUM_FEATURES)
        pos_encoder = FrequencyPositionalEncoding3d(d_model=attention_cfg.NUM_FEATURES, max_spatial_shape=torch.IntTensor([2, 2, 2]), dropout=0.)
        encoder = TransformerEncoder(attention_cfg, pos_encoder)
        self.src_key_padding_mask_eval(input, encoder)
    def test_src_key_padding_mask_feed_forward(self):
        attention_cfg = EasyDict(dict(NUM_FEATURES=6, NUM_HEADS=1, NUM_HIDDEN_FEATURES=16, NUM_LAYERS=1, DROPOUT=0.))
        d_input = 3
        input = torch.randn(10, 27, attention_cfg.NUM_FEATURES)
        pos_encoder = FeedForwardPositionalEncoding(d_input=d_input, d_output=attention_cfg.NUM_FEATURES)
        # Need to set to eval because of batch norm
        pos_encoder.eval()
        encoder = TransformerEncoder(attention_cfg, pos_encoder)
        input_point_locations = torch.randn(10, 27, d_input)
        self.src_key_padding_mask_eval(input, encoder, input_point_locations)


class TestFrequencyPositionalEncoding3d(unittest.TestCase):
    def test_frequency_positional_encoding_3d_asserts(self):
        with self.assertRaises(AssertionError):
            FrequencyPositionalEncoding3d(11, max_spatial_shape=[3, 3, 3])
        with self.assertRaises(AssertionError):
            FrequencyPositionalEncoding3d(12, max_spatial_shape=[3, 3])
        with self.assertRaises(AssertionError):
            FrequencyPositionalEncoding3d(11, max_spatial_shape=[3, 3, 3, 3])

    def test_frequency_positional_encoding_3d_default(self):
        fpe = FrequencyPositionalEncoding3d(6, max_spatial_shape=torch.IntTensor([2, 2, 2]), dropout=0)

        # Check if unique positional encoding is the same for all dimensions
        x_unique = fpe.pe[:2,...].view(2,-1).unique(dim=1)
        y_unique = fpe.pe[2:4,...].view(2,-1).unique(dim=1)
        z_unique = fpe.pe[4:6,...].view(2,-1).unique(dim=1)

        self.assertTrue(torch.all(torch.isclose(x_unique, y_unique)))
        self.assertTrue(torch.all(torch.isclose(y_unique, z_unique)))

        # Features should be the same at the same index
        indices = [0, 1]
        same_features = fpe.pe[:, indices, indices, indices]
        self.assertTrue(torch.all(torch.isclose(same_features[:2, ...], same_features[2:4, ...])))
        # self.assertTrue(torch.all(torch.isclose(same_features[2:4, ...], same_features[4:6, ...])))

        # Check regular forward
        input = torch.randn(2, 2, 2, 2, 6)
        output = fpe(input.view(2, -1, 6), positional_input=None)
        output = output.view(2, 2, 2, 2, 6)
        self.assertTrue(torch.all(torch.isclose(input + fpe.pe.permute(1, 2, 3, 0).unsqueeze(0), output)))

        # Check different grid_size
        bigger_fpe = FrequencyPositionalEncoding3d(12, max_spatial_shape=torch.IntTensor([3, 3, 3]), dropout=0)
        smaller_input = torch.zeros(10, 2, 3, 2, 12)
        grid_size = torch.IntTensor([2, 3, 2])
        smaller_output = bigger_fpe(smaller_input.view(10, -1, 12), positional_input=None, grid_size=grid_size)
        smaller_output = smaller_output.view(10, 2, 3, 2, 12)
        self.assertTrue(torch.all(torch.isclose(smaller_input + bigger_fpe.pe.permute(1, 2, 3, 0)[:2, :3, :2, :].unsqueeze(0), smaller_output)))

        # Check that each output map is the same for each batch
        zero_input = torch.zeros(10, 8, 6)
        zero_output = fpe(zero_input, positional_input=None)
        avg_zero_input = torch.mean(zero_output, dim=0)
        self.assertTrue(torch.all(torch.isclose(avg_zero_input, zero_output[0, ...])))


class TestFeedForwardPositionalEncoding(unittest.TestCase):
    def test_feed_forward_positional_encoding_default(self):
        d_model = 6
        d_input = 3
        input = torch.randn(10, 27, d_model)

        # Check if output is the exact same shape as input
        ffn = FeedForwardPositionalEncoding(d_input=d_input, d_output=d_model)
        input_locations = torch.randn(10, 27, d_input)
        output = ffn(input, input_locations)
        self.assertTrue(torch.all(torch.eq(torch.tensor(input.shape), torch.tensor(output.shape))))
