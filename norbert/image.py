import numpy as np
from PIL import Image
import tempfile as tmp
import piexif
import piexif.helper
import json

eps = np.finfo(np.float).eps


class ImageEncoder(object):
    def __init__(
        self,
        format='jpg',
        quality=75,
        qtable=None
    ):
        self.format = format
        self.quality = quality

        if qtable is not None:
            self.qtables = [qtable]
        else:
            self.qtables = None

    def encodedecode(self, X):
        """encode/decode on the fly"""
        image_file = tmp.NamedTemporaryFile(suffix='.' + self.format)
        y = self.decode(
            self.encode(X, out=image_file.name)
        )
        image_file.close()
        return y

    def encode(self, X, out=None, user_comment_dict=None):
        if out is not None:
            buf = np.flipud(np.squeeze(X).T)
            if buf.ndim <= 2:
                img = Image.fromarray(buf, 'L')
            else:
                buf = np.vstack((buf, np.zeros((1,) + buf[0].shape))).T
                # buf = buf[[2, 0, 1], ...]
                img = Image.fromarray(buf.astype(np.int8), 'RGB')

            if user_comment_dict is not None:
                user_comment = piexif.helper.UserComment.dump(
                    json.dumps(user_comment_dict)
                )
                exif_ifd = {
                    piexif.ExifIFD.UserComment: user_comment,
                }
                exif_dict = {"Exif": exif_ifd}
                exif_bytes = piexif.dump(exif_dict)
                img.save(
                    out,
                    quality=self.quality,
                    optimize=True,
                    exif=exif_bytes,
                    qtables=self.qtables
                )
            else:
                img.save(
                    out,
                    quality=self.quality,
                    optimize=True,
                    qtables=self.qtables
                )

    def decode(self, buf):
        img = Image.open(buf)
        img = np.array(img).astype(np.uint8)
        if img.ndim <= 2:
            return np.atleast_3d(img).transpose((1, 0, 2))
        else:
            return img[:, :, [1, 2]]
