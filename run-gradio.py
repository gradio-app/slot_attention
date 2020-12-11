
from absl import logging
import tensorflow as tf
import numpy as np
import gradio as gr
import PIL
import os

# When executing this notebook out of a subfolder, use the command below to
# change to the project's root folder (required for imports):
# %cd ..

import slot_attention.model as model_utils
import subprocess

# Hyperparameters.
seed = 0
batch_size = 1
num_slots = 7
num_iterations = 3
resolution = (128, 128)
ckpt_path = "/tmp/object_discovery/"  # Path to model checkpoint.

if not os.path.exists(ckpt_path):
    os.mkdir(ckpt_path)
import boto3
s3=boto3.client('s3')
list=s3.list_objects(Bucket='gradio-slot')["Contents"]
for key in list:
    s3.download_file("gradio-slot", key["Key"], os.path.join("/tmp", key["Key"]))
print("-> DONE")

def load_model(checkpoint_dir, num_slots=11, num_iters=3, batch_size=16):
    resolution = (128, 128)
    model = model_utils.build_model(
        resolution, batch_size, num_slots, num_iters,
        model_type="object_discovery")

    ckpt = tf.train.Checkpoint(network=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=checkpoint_dir, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logging.info("Restored from %s", ckpt_manager.latest_checkpoint)

    return model

model = load_model(ckpt_path, num_slots=num_slots, num_iters=num_iterations,
                   batch_size=batch_size)


def renormalize(x):
  """Renormalize from [-1, 1] to [0, 1]."""
  return x / 2. + 0.5

def get_prediction(model, batch, idx=0):
  recon_combined, recons, masks, slots = model(batch["image"])
  image = renormalize(batch["image"])[idx]
  recon_combined = renormalize(recon_combined)[idx]
  recons = renormalize(recons)[idx]
  masks = masks[idx]
  return image, recon_combined, recons, masks, slots


def show_slots(img):
    img = img / 128. - 1
    batch = {
        "image": tf.convert_to_tensor(np.expand_dims(img, axis=0))
    }
    image, recon_combined, recons, masks, slots = get_prediction(model, batch)
    mask_images = []
    for i in range(num_slots):
        mask_images.append(np.array(recons[i] * masks[i] + (1 - masks[i])))
    mask_images = [PIL.Image.fromarray((mask_image * 255).astype(np.uint8)) for mask_image in mask_images]
    mask_images[0].save('out.gif', format='GIF', append_images=mask_images[1:], save_all=True, duration=1000, loop=0)
    recon_combined = np.array(recon_combined)
    recon_combined = np.clip(recon_combined, -1., 1.)

    return recon_combined, "out.gif"


iface = gr.Interface(
    show_slots, 
    gr.inputs.Image(shape=(128, 128)), 
    [
        gr.outputs.Image(label="Reconstructed"),
        gr.outputs.Image(label="Masks")
    ],
    examples=[
        ["examples/" + img] for img in os.listdir("examples/")
    ],
    examples_per_page=4,
    title="Slot Attention",
    description="This is an interface implementation for 'Object-Centric Learning with Slot Attention', trained on the CLVER dataset. Click the examples to load them or upload your own images from the CLVER dataset.",
    article="""This is on based the paper ["Object-Centric Learning with Slot Attention"](https://arxiv.org/abs/2006.15055) by Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf.
    """,
    layout="unaligned",
)

if __name__ == "__main__":
    iface.launch()