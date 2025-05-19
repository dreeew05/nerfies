# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A module to help create embeddings in Jax."""
from flax import linen as nn
import jax.numpy as jnp

from nerfies import types


class GloEncoder(nn.Module):
    """A GLO encoder module, which is just a thin wrapper around nn.Embed.

    Attributes:
      num_embeddings: The number of embeddings.
      features: The dimensions of each embedding.
      embedding_init: The initializer to use for each.
    """

    num_embeddings: int
    features: int
    embedding_init: types.Activation = nn.initializers.uniform(scale=0.05)

    def setup(self):
        self.embed = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            embedding_init=self.embedding_init,
        )

    # Inside nerfies/glo.py, within GloEncoder class


def __call__(
    self, inputs: jnp.ndarray
) -> jnp.ndarray:  # Inputs expected to be JAX array indices
    print(f"--- Entering GloEncoder.__call__ ---")
    print(
        f"inputs type: {type(inputs)}, shape: {inputs.shape if hasattr(inputs, 'shape') else 'N/A'}"
    )

    if inputs.shape[-1] == 1:
        inputs = jnp.squeeze(inputs, axis=-1)

    # Call to nn.Embed happens here
    print("Calling nn.Embed...")
    result = self.embed(inputs)
    print("nn.Embed called.")

    print(f"--- Exiting GloEncoder.__call__ ---")
    return result
