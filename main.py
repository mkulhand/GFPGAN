from fastapi import FastAPI, Response, Request
from pydantic import BaseModel
import base64, tempfile, subprocess, uuid, shutil, os, json


app = FastAPI()

os.makedirs("tmp", exist_ok=True)

class Upscaler:
    tmp: str
    input_path: str

    def __init__(self, input_path: str):
        temp_id = uuid.uuid1()
        self.tmp = f"tmp/{temp_id}"
        self.input_path = input_path
        os.makedirs(f"{self.tmp}/frames", exist_ok=True)
        os.makedirs(f"{self.tmp}/restored_frames", exist_ok=True)
        os.makedirs(f"{self.tmp}/output", exist_ok=True)

    def split_frames(self) -> None:
        cmd = ["ffmpeg", "-i", self.input_path, f"{self.tmp}/frames/frame_%04d.png"]

        subprocess.run(cmd, check=True)

    def run_upscale(self) -> None:
        cmd = [
            "python3",
            "inference_gfpgan.py",
            "-i",
            f"{self.tmp}/frames",
            "-o",
            f"{self.tmp}/restored_frames",
            "-v",
            "1.4",
            "-s",
            "2",
            "-p",
            "3",
            "--min_face_size",
            "40",
            "--max_face_size",
            "150",
        ]

        subprocess.run(cmd, check=True)

    def restore_video(self) -> None:
        cmd = [
            "ffmpeg",
            "-r",
            "32",
            "-i",
            f"{self.tmp}/restored_frames/frame_%04d.png",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "2.5M",
            "-maxrate",
            "10M",
            "-minrate",
            "5M",
            "-crf",
            "10",
            "-cpu-used",
            "2",
            "-row-mt",
            "1",
            "-threads",
            "8",
            f"{self.tmp}/output/output.webm",
        ]

        subprocess.run(cmd, check=True)

    def webm_to_base64(self) -> str:
        try:
            with open(f"{self.tmp}/output/output.webm", "rb") as video_file:
                video_data = video_file.read()
                base64_string = base64.b64encode(video_data).decode("utf-8")
                return base64_string
        except FileNotFoundError:
            print(f"File {self.tmp}/output/output.webm not found")
            return None
        except Exception as e:
            print(f"Error converting to base64: {e}")
            return None

    def clean_tmp(self):
        try:
            shutil.rmtree(self.tmp)
            print(f"Successfully removed folder: {self.tmp}")
        except FileNotFoundError:
            print(f"Folder not found: {self.tmp}")
        except PermissionError:
            print(f"Permission denied: {self.tmp}")
        except Exception as e:
            print(f"Error removing folder: {e}")

    def upscale(self) -> str:
        self.split_frames()
        self.run_upscale()
        self.restore_video()
        output_base64 = self.webm_to_base64()
        self.clean_tmp()

        return output_base64


class UpscaleInput(BaseModel):
    b64: str


@app.post("/upscale")
def upscale_vid(request: Request, input: UpscaleInput):
    if request.headers.get("Authorization") != f"Bearer {os.environ.get("GFPGAN_API")}":
        return Response(status_code=401)

    video_data = base64.b64decode(input.b64)

    # Create temporary file for the original video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_data)
        tmp.flush()
        file_path = tmp.name
        output_base64 = Upscaler(file_path).upscale()

    return Response(content=json.dumps({"base64": output_base64}))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
