import os
import glob
from .bases import BaseImageDataset

def _parse_camid_from_name(filename: str) -> int:
    """
    예: 1_001_1_01.png -> cam = 1 (0-index로 0)
        1_001_2_06.png -> cam = 2 (0-index로 1)
    실패하면 0 반환.
    """
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) >= 3:
        try:
            cam = int(parts[2])  # 3번째 토큰
            return max(cam - 1, 0)
        except ValueError:
            return 0
    return 0


class _CUHK03_Base(BaseImageDataset):
    """
    공통 로직: pid 폴더 구조 기반
    train/query/gallery에서 pid는 폴더명으로 결정.
    camid는 파일명에서 파싱(권장).
    """
    dataset_dir = None  # 하위 클래스에서 지정 (예: 'CUHK03/detected')

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super().__init__()
        if self.dataset_dir is None:
            raise ValueError("dataset_dir must be set in subclass")

        self.dataset_dir = os.path.join(root, self.dataset_dir)

        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.query_dir = os.path.join(self.dataset_dir, "test", "query")
        self.gallery_dir = os.path.join(self.dataset_dir, "test", "gallery")

        self._check_before_run()

        self.pid_begin = pid_begin

        train = self._process_pid_folders(self.train_dir, relabel=True)
        query = self._process_pid_folders(self.query_dir, relabel=False)
        gallery = self._process_pid_folders(self.gallery_dir, relabel=False)

        if verbose:
            print(f"=> {self.__class__.__name__} loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not os.path.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available")
        if not os.path.exists(self.query_dir):
            raise RuntimeError(f"'{self.query_dir}' is not available")
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError(f"'{self.gallery_dir}' is not available")

    def _process_pid_folders(self, folder, relabel: bool):
        pid_dirs = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
        # pid 폴더명이 '0001' 같은 문자열이므로 int 변환해서 정렬 안정화
        pid_ints = []
        for d in pid_dirs:
            try:
                pid_ints.append(int(d))
            except ValueError:
                # 숫자 아닌 폴더는 무시
                pass
        pid_ints = sorted(pid_ints)
        pid2label = {pid: idx for idx, pid in enumerate(pid_ints)}

        dataset = []
        for pid in pid_ints:
            pid_path = os.path.join(folder, f"{pid:04d}") if os.path.isdir(os.path.join(folder, f"{pid:04d}")) else os.path.join(folder, str(pid))
            if not os.path.isdir(pid_path):
                continue

            new_pid = pid2label[pid] if relabel else pid
            # 확장자: png/jpg/jpeg 모두 허용
            img_paths = []
            img_paths += glob.glob(os.path.join(pid_path, "*.png"))
            img_paths += glob.glob(os.path.join(pid_path, "*.jpg"))
            img_paths += glob.glob(os.path.join(pid_path, "*.jpeg"))

            for img_path in sorted(img_paths):
                camid = _parse_camid_from_name(img_path)  # 0-index camid
                dataset.append((img_path, self.pid_begin + new_pid, camid, 1))

        return dataset


class CUHK03_Detected(_CUHK03_Base):
    # 네가 말한 구조: CUHK폴더 안에 detected/train, detected/test/query, detected/test/gallery
    dataset_dir = os.path.join("CUHK03", "detected")


class CUHK03_Labeled(_CUHK03_Base):
    dataset_dir = os.path.join("CUHK03", "labeled")