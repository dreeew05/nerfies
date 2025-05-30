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

"""Different model implementation plus a general port for all the models."""
from typing import Any, Dict, Mapping, Optional, Tuple, Sequence

from flax import linen as nn
import immutabledict
from jax import random
import jax.numpy as jnp

from nerfies import configs
from nerfies import glo
from nerfies import model_utils
from nerfies import modules
from nerfies import types
from nerfies import warping


class NerfModel(nn.Module):
    """Nerf NN Model with both coarse and fine MLPs.

    Attributes:
      num_coarse_samples: int, the number of samples for coarse nerf.
      num_fine_samples: int, the number of samples for fine nerf.
      use_viewdirs: bool, use viewdirs as a condition.
      near: float, near clip.
      far: float, far clip.
      noise_std: float, std dev of noise added to regularize sigma output.
      nerf_trunk_depth: int, the depth of the first part of MLP.
      nerf_trunk_width: int, the width of the first part of MLP.
      nerf_rgb_branch_depth: int, the depth of the second part of MLP.
      nerf_rgb_branch_width: int, the width of the second part of MLP.
      activation: the activation function used in the MLP.
      sigma_activation: the activation function applied to the sigma density.
      nerf_skips: which layers to add skip layers in the NeRF model.
      alpha_channels: int, the number of alpha_channelss.
      rgb_channels: int, the number of rgb_channelss.
      use_stratified_sampling: use stratified sampling.
      use_white_background: composite rendering on to a white background.
      num_nerf_point_freqs: degree of positional encoding for positions.
      num_nerf_viewdir_freqs: degree of positional encoding for viewdirs.
      use_linear_disparity: sample linearly in disparity rather than depth.
      num_appearance_embeddings: the number of appearance exemplars.
      num_appearance_features: the dimension size for the appearance code.
      num_camera_embeddings: the number of camera exemplars.
      num_camera_features: the dimension size for the camera code.
      num_warp_freqs: the number of frequencies for the warp positional encoding.
      num_warp_embeddings: the number of embeddings for the warp GLO encoder.
      num_warp_features: the number of features for the warp GLO encoder.
      use_appearance_metadata: if True use the appearance metadata.
      use_camera_metadata: if True use the camera metadata.
      use_warp: whether to use the warp field or not.
      use_warp_jacobian: if True the model computes and returns the Jacobian of
        the warped points.
      use_weights: if True return the density weights.
      use_trunk_condition: whether to feed the appearance metadata to the trunk.
      use_alpha_condition: whether to feed the appearance metadata to the alpha
        branch.
      use_rgb_condition: whether to feed the appearance metadata to the rgb
        branch.
      warp_kwargs: extra keyword arguments for the warp field.
    """

    num_coarse_samples: int
    num_fine_samples: int
    use_viewdirs: bool
    near: float
    far: float
    noise_std: Optional[float]
    nerf_trunk_depth: int
    nerf_trunk_width: int
    nerf_rgb_branch_depth: int
    nerf_rgb_branch_width: int
    nerf_skips: Tuple[int]
    alpha_channels: int
    rgb_channels: int
    use_stratified_sampling: bool
    num_nerf_point_freqs: int
    num_nerf_viewdir_freqs: int

    appearance_ids: Sequence[int]
    camera_ids: Sequence[int]
    warp_ids: Sequence[int]

    num_appearance_features: int
    num_camera_features: int
    num_warp_features: int
    num_warp_freqs: int

    appearance_ids: Sequence[int]
    camera_ids: Sequence[int]
    warp_ids: Sequence[int]

    activation: types.Activation = nn.relu
    sigma_activation: types.Activation = nn.relu
    use_white_background: bool = False
    use_linear_disparity: bool = False
    use_sample_at_infinity: bool = True

    warp_field_type: str = "se3"
    warp_metadata_encoder_type: str = "glo"
    use_appearance_metadata: bool = False
    use_camera_metadata: bool = False
    use_warp: bool = False
    use_warp_jacobian: bool = False
    use_weights: bool = False
    use_trunk_condition: bool = False
    use_alpha_condition: bool = False
    use_rgb_condition: bool = False
    warp_kwargs: Mapping[str, Any] = immutabledict.immutabledict()

    metadata_encoded: bool = False

    @property
    def num_appearance_embeddings(self):
        return max(self.appearance_ids) + 1

    @property
    def num_warp_embeddings(self):
        return max(self.warp_ids) + 1

    @property
    def num_camera_embeddings(self):
        return max(self.camera_ids) + 1

    @staticmethod
    def create_warp_field(model, num_batch_dims):
        return warping.create_warp_field(
            field_type=model.warp_field_type,
            num_freqs=model.num_warp_freqs,
            num_embeddings=model.num_warp_embeddings,
            num_features=model.num_warp_features,
            num_batch_dims=num_batch_dims,
            metadata_encoder_type=model.warp_metadata_encoder_type,
            **model.warp_kwargs,
        )

    # Inside nerfies/models.py, within the NerfModel class definition

    def setup(self):
        print(f"--- Entering NerfModel.setup ---")
        print(f"Original self.appearance_ids type: {type(self.appearance_ids)}")
        print(f"Original self.camera_ids type: {type(self.camera_ids)}")
        print(f"Original self.warp_ids type: {type(self.warp_ids)}")

        # Ensure jnp is imported in models.py (add at the top if not there)
        import jax.numpy as jnp

        # --- Fix for SetAttributeInModuleSetupError ---
        # Create NEW internal attributes for the JAX array versions.
        # You cannot reassign self.appearance_ids etc. because they are frozen
        # attributes set during __init__.
        self._appearance_ids_jnp = jnp.array(self.appearance_ids)
        self._camera_ids_jnp = jnp.array(self.camera_ids)
        self._warp_ids_jnp = jnp.array(self.warp_ids)
        # --- End of Fix ---

        # --- Debugging Prints (optional, but helpful) ---
        print(f"Created _appearance_ids_jnp type: {type(self._appearance_ids_jnp)}")
        print(f"Created _camera_ids_jnp type: {type(self._camera_ids_jnp)}")
        print(f"Created _warp_ids_jnp type: {type(self._warp_ids_jnp)}")
        print(f"--- After creating new JAX array attributes in NerfModel.setup ---")
        # --- End of Debugging Prints ---

        # The rest of the original setup method continues here,
        # initializing submodules.

        if self.use_warp:
            print("Initializing warp_field...")
            # --- IMPORTANT ---
            # If create_warp_field (in warping.py) or the warp field modules
            # it creates internally access the parent model's attributes
            # like model.warp_ids, they will still access the ORIGINAL tuple attribute.
            # These parts of the code (in warping.py) NEED to be updated to use
            # the new self._warp_ids_jnp attribute if they use the IDs in a JAX operation
            # expecting an array.
            # ---------------
            self.warp_field = self.create_warp_field(self, num_batch_dims=2)
            print("Warp_field initialized.")

        self.point_encoder = model_utils.vmap_module(
            modules.SinusoidalEncoder, num_batch_dims=2
        )(num_freqs=self.num_nerf_point_freqs)
        self.viewdir_encoder = model_utils.vmap_module(
            modules.SinusoidalEncoder, num_batch_dims=1
        )(num_freqs=self.num_nerf_viewdir_freqs)

        if self.use_appearance_metadata:
            # --- IMPORTANT ---
            # The GloEncoder constructor uses num_embeddings (integer derived from max(tuple)+1).
            # This is fine. BUT, if GloEncoder's *internal* setup or __call__ method
            # accesses the parent model's attributes (like appearance_ids)
            # passed to it implicitly via the module hierarchy, it must be updated
            # to use the new self._appearance_ids_jnp attribute for JAX operations.
            # ---------------
            self.appearance_encoder = glo.GloEncoder(
                num_embeddings=self.num_appearance_embeddings,  # Uses max(original tuple)+1
                features=self.num_appearance_features,
            )
        if self.use_camera_metadata:
            # --- IMPORTANT ---
            # Same applies to camera_encoder with self._camera_ids_jnp
            # ---------------
            self.camera_encoder = glo.GloEncoder(
                num_embeddings=self.num_camera_embeddings,
                features=self.num_camera_features,
            )

        nerf_mlps = {
            "coarse": modules.NerfMLP(
                trunk_depth=self.nerf_trunk_depth,
                trunk_width=self.nerf_trunk_width,
                rgb_branch_depth=self.nerf_rgb_branch_depth,
                rgb_branch_width=self.nerf_rgb_branch_width,
                activation=self.activation,
                skips=self.nerf_skips,
                alpha_channels=self.alpha_channels,
                rgb_channels=self.rgb_channels,
            )
        }
        if self.num_fine_samples > 0:
            nerf_mlps["fine"] = modules.NerfMLP(
                trunk_depth=self.nerf_trunk_depth,
                trunk_width=self.nerf_trunk_width,
                rgb_branch_depth=self.nerf_rgb_branch_depth,
                rgb_branch_width=self.nerf_rgb_branch_width,
                activation=self.activation,
                skips=self.nerf_skips,
                alpha_channels=self.alpha_channels,
                rgb_channels=self.rgb_channels,
            )
        self.nerf_mlps = nerf_mlps

        print(f"--- Exiting NerfModel.setup ---")

    # --- IMPORTANT: Update other methods in NerfModel ---
    # Any method in NerfModel (e.g., __call__, render_samples, get_condition_inputs)
    # or any other function/method that receives the NerfModel instance and uses
    # self.appearance_ids, self.camera_ids, or self.warp_ids in a JAX operation
    # expecting an array MUST be updated to use the new self._appearance_ids_jnp,
    # self._camera_ids_jnp, self._warp_ids_jnp attributes instead.
    # You will likely need to find these specific usages throughout models.py,
    # warping.py, and potentially other files if they receive the model instance
    # or parts of its state and access these attributes.
    # --- End of IMPORTANT ---

    def get_condition_inputs(self, viewdirs, metadata, metadata_encoded=False):
        """Create the condition inputs for the NeRF template."""
        trunk_conditions = []
        alpha_conditions = []
        rgb_conditions = []

        # Point attribute predictions
        if self.use_viewdirs:
            viewdirs_embed = self.viewdir_encoder(viewdirs)
            rgb_conditions.append(viewdirs_embed)

        if self.use_appearance_metadata:
            if metadata_encoded:
                appearance_code = metadata["appearance"]
            else:
                appearance_code = self.appearance_encoder(metadata["appearance"])
            if self.use_trunk_condition:
                trunk_conditions.append(appearance_code)
            if self.use_alpha_condition:
                alpha_conditions.append(appearance_code)
            if self.use_alpha_condition:
                rgb_conditions.append(appearance_code)

        if self.use_camera_metadata:
            if metadata_encoded:
                camera_code = metadata["camera"]
            else:
                camera_code = self.camera_encoder(metadata["camera"])
            rgb_conditions.append(camera_code)

        # The condition inputs have a shape of (B, C) now rather than (B, S, C)
        # since we assume all samples have the same condition input. We might want
        # to change this later.
        trunk_conditions = (
            jnp.concatenate(trunk_conditions, axis=-1) if trunk_conditions else None
        )
        alpha_conditions = (
            jnp.concatenate(alpha_conditions, axis=-1) if alpha_conditions else None
        )
        rgb_conditions = (
            jnp.concatenate(rgb_conditions, axis=-1) if rgb_conditions else None
        )
        return trunk_conditions, alpha_conditions, rgb_conditions

    def render_samples(
        self,
        level,
        points,
        z_vals,
        directions,
        viewdirs,
        metadata,
        warp_extra,
        use_warp=True,
        use_warp_jacobian=False,
        metadata_encoded=False,
        return_points=False,
        return_weights=False,
    ):

        print(f"--- Entering NerfModel.render_samples (level: {level}) ---")
        # Add checks for inputs if needed, but during init they should be JAX arrays
        # print(f"metadata type: {type(metadata)}") # metadata is a dict
        # print(f"metadata['warp'] type: {type(metadata.get('warp'))}") # Check type of specific metadata entries

        trunk_condition, alpha_condition, rgb_condition = self.get_condition_inputs(
            viewdirs, metadata, metadata_encoded
        )
        # Add checks for conditions if needed
        # print(f"trunk_condition type: {type(trunk_condition)}") # Could be None or JAX array

        out = {}
        if return_points:
            out["points"] = points
        # Apply the deformation field to the samples.
        if use_warp:
            metadata_channels = self.num_warp_features if metadata_encoded else 1
            warp_metadata = (
                metadata["time"]
                if self.warp_metadata_encoder_type == "time"
                else metadata["warp"]
            )
            # This is the suspicious broadcast_to call location from earlier
            print(f"warp_metadata before broadcast_to type: {type(warp_metadata)}")
            print(
                f"warp_metadata shape: {warp_metadata.shape if hasattr(warp_metadata, 'shape') else 'N/A'}"
            )
            # ... other print checks on warp_metadata[:, jnp.newaxis, :] if needed

            warp_metadata = jnp.broadcast_to(  # <--- Error might originate here
                warp_metadata[:, jnp.newaxis, :],
                shape=(*points.shape[:2], metadata_channels),
            )
            print(
                f"warp_metadata after broadcast_to type: {type(warp_metadata)}"
            )  # Should be JAX array

            # Call to warp_field.__call__ happens here
            print("Calling warp_field...")
            warp_out = self.warp_field(
                points, warp_metadata, warp_extra, use_warp_jacobian, metadata_encoded
            )
            print("Warp_field called.")
            # ... rest of the method
        # ... rest of render_samples
        print(f"--- Exiting NerfModel.render_samples ---")
        return out

    def __call__(  # This method is traced during model.init
        self,
        rays_dict: Dict[str, Any],
        warp_extra: Dict[str, Any],
        metadata_encoded=False,
        use_warp=True,
        return_points=False,
        return_weights=False,
        return_warp_jacobian=False,
        deterministic=False,
    ):

        print(f"--- Entering NerfModel.__call__ ---")
        # Add checks for inputs received during init
        # print(f"rays_dict type: {type(rays_dict)}") # Should be dict
        # print(f"rays_dict['metadata'] type: {type(rays_dict.get('metadata'))}") # Should be dict
        # print(f"rays_dict['metadata']['appearance'] type: {type(rays_dict['metadata'].get('appearance'))}") # Should be JAX array during init

        use_warp = self.use_warp and use_warp
        return_weights = self.use_weights or return_weights
        origins = rays_dict["origins"]
        directions = rays_dict["directions"]
        metadata = rays_dict["metadata"]
        if "viewdirs" in rays_dict:
            viewdirs = rays_dict["viewdirs"]
        else:
            viewdirs = directions

        # Evaluate coarse samples.
        print("Sampling coarse rays...")
        z_vals, points = (
            model_utils.sample_along_rays(  # Error could be here too if inputs are wrong type
                self.make_rng("coarse"),
                origins,
                directions,
                self.num_coarse_samples,  # This is an integer, fine
                self.near,  # Float, fine
                self.far,  # Float, fine
                self.use_stratified_sampling,  # Bool, fine
                self.use_linear_disparity,  # Bool, fine
            )
        )
        print("Coarse rays sampled.")
        print("Rendering coarse samples...")
        coarse_ret = self.render_samples(  # Calls render_samples
            "coarse",
            points,
            z_vals,
            directions,
            viewdirs,
            metadata,
            warp_extra,
            use_warp=use_warp,
            use_warp_jacobian=return_warp_jacobian or self.use_warp_jacobian,
            metadata_encoded=metadata_encoded,
            return_points=return_points,
            return_weights=True,
        )
        print("Coarse samples rendered.")
        out = {"coarse": coarse_ret}

        # Evaluate fine samples.
        if self.num_fine_samples > 0:
            print("Sampling fine rays...")
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_vals, points = model_utils.sample_pdf(  # Error could be here too
                self.make_rng("fine"),
                z_vals_mid,
                coarse_ret["weights"][..., 1:-1],
                origins,
                directions,
                self.num_fine_samples,  # Integer, fine
                self.use_stratified_sampling,  # Bool, fine
            )
            print("Fine rays sampled.")
            print("Rendering fine samples...")
            out["fine"] = self.render_samples(  # Calls render_samples again
                "fine",
                points,
                z_vals,
                directions,
                viewdirs,
                metadata,
                warp_extra,
                use_warp=use_warp,
                use_warp_jacobian=return_warp_jacobian,
                metadata_encoded=metadata_encoded,
                return_points=return_points,
                return_weights=return_weights,
            )
            print("Fine samples rendered.")

        if not return_weights:
            if "weights" in out["coarse"]:
                del out["coarse"]["weights"]
            if "fine" in out and "weights" in out["fine"]:
                del out["fine"]["weights"]

        print(f"--- Exiting NerfModel.__call__ ---")
        return out


def construct_nerf(
    key,
    config: configs.ModelConfig,
    batch_size: int,
    appearance_ids: Sequence[int],
    camera_ids: Sequence[int],
    warp_ids: Sequence[int],
    near: float,
    far: float,
    use_warp_jacobian: bool = False,
    use_weights: bool = False,
):
    """Neural Randiance Field.

    Args:
      key: jnp.ndarray. Random number generator.
      config: model configs.
      batch_size: the evaluation batch size used for shape inference.
      appearance_ids: the appearance embeddings.
      camera_ids: the camera embeddings.
      warp_ids: the warp embeddings.
      near: the near plane of the scene.
      far: the far plane of the scene.
      use_warp_jacobian: if True the model computes and returns the Jacobian of
        the warped points.
      use_weights: if True return the density weights from the NeRF.

    Returns:
      model: nn.Model. Nerf model with parameters.
      state: flax.Module.state. Nerf model state for stateful parameters.
    """
    num_nerf_point_freqs = config.num_nerf_point_freqs
    num_nerf_viewdir_freqs = config.num_nerf_viewdir_freqs
    num_coarse_samples = config.num_coarse_samples
    num_fine_samples = config.num_fine_samples
    use_viewdirs = config.use_viewdirs
    noise_std = config.noise_std
    use_stratified_sampling = config.use_stratified_sampling
    use_white_background = config.use_white_background
    nerf_trunk_depth = config.nerf_trunk_depth
    nerf_trunk_width = config.nerf_trunk_width
    nerf_rgb_branch_depth = config.nerf_rgb_branch_depth
    nerf_rgb_branch_width = config.nerf_rgb_branch_width
    nerf_skips = config.nerf_skips
    alpha_channels = config.alpha_channels
    rgb_channels = config.rgb_channels
    use_linear_disparity = config.use_linear_disparity

    model = NerfModel(
        num_coarse_samples=num_coarse_samples,
        num_fine_samples=num_fine_samples,
        use_viewdirs=use_viewdirs,
        near=near,
        far=far,
        noise_std=noise_std,
        nerf_trunk_depth=nerf_trunk_depth,
        nerf_trunk_width=nerf_trunk_width,
        nerf_rgb_branch_depth=nerf_rgb_branch_depth,
        nerf_rgb_branch_width=nerf_rgb_branch_width,
        use_alpha_condition=config.use_alpha_condition,
        use_rgb_condition=config.use_rgb_condition,
        activation=config.activation,
        sigma_activation=config.sigma_activation,
        nerf_skips=nerf_skips,
        alpha_channels=alpha_channels,
        rgb_channels=rgb_channels,
        use_stratified_sampling=use_stratified_sampling,
        use_white_background=use_white_background,
        use_sample_at_infinity=config.use_sample_at_infinity,
        num_nerf_point_freqs=num_nerf_point_freqs,
        num_nerf_viewdir_freqs=num_nerf_viewdir_freqs,
        use_linear_disparity=use_linear_disparity,
        use_warp_jacobian=use_warp_jacobian,
        use_weights=use_weights,
        use_appearance_metadata=config.use_appearance_metadata,
        use_camera_metadata=config.use_camera_metadata,
        use_warp=config.use_warp,
        appearance_ids=appearance_ids,
        camera_ids=camera_ids,
        warp_ids=warp_ids,
        num_appearance_features=config.appearance_metadata_dims,
        num_camera_features=config.camera_metadata_dims,
        num_warp_freqs=config.num_warp_freqs,
        num_warp_features=config.num_warp_features,
        warp_field_type=config.warp_field_type,
        warp_metadata_encoder_type=config.warp_metadata_encoder_type,
        warp_kwargs=immutabledict.immutabledict(config.warp_kwargs),
    )

    init_rays_dict = {
        "origins": jnp.ones((batch_size, 3), jnp.float32),
        "directions": jnp.ones((batch_size, 3), jnp.float32),
        "metadata": {
            "warp": jnp.ones((batch_size, 1), jnp.uint32),
            "camera": jnp.ones((batch_size, 1), jnp.uint32),
            "appearance": jnp.ones((batch_size, 1), jnp.uint32),
            "time": jnp.ones((batch_size, 1), jnp.float32),
        },
    }
    warp_extra = {
        "alpha": 0.0,
        "time_alpha": 0.0,
    }

    key, key1, key2 = random.split(key, 3)
    params = model.init(
        {"params": key, "coarse": key1, "fine": key2},
        init_rays_dict,
        warp_extra=warp_extra,
    )["params"]

    return model, params
