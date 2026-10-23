class PTrack:
    last_id = 0
    total_count = 0

    def __init__(self, frames_data: list, track_id=None):
        self.track_id = track_id if track_id else self.get_new_track_id()
        if self.track_id > PTrack.last_id:
            PTrack.last_id = self.track_id
        assert len(frames_data) != 0, f"Данные о бибкосах для трека {self.track_id} пустые"
        self.frames_data = frames_data
        self.is_solved = False
        self.final_code = None

    @staticmethod
    def get_new_track_id():
        PTrack.last_id += 1
        return PTrack.last_id - 1

    def check_solved(self) -> bool:
        most_probable = dict()
        for frame_data in self.frames_data:
            if frame_data is None:
                continue
            if frame_data["code"] not in most_probable:
                most_probable[frame_data["code"]] = 1
            else:
                most_probable[frame_data["code"]] += 1
        self.is_solved = True if len(most_probable) == 1 else False
        self.final_code = list(most_probable.keys())[0] if self.is_solved else None
        return self.is_solved

    def set_code(self, code: int):
        for frame_data in self.frames_data:
            if frame_data is not None:
                frame_data["code"] = code
        self.is_solved = True

    def set_code_till_now(self, code: int, frame_num: int):
        for num, frame_data in enumerate(self.frames_data):
            if num > frame_num:
                return
            if frame_data is not None:
                frame_data["code"] = code
        self.check_solved()

    def remove_track(self, tracks: dict, frame_storage: list[list]):
        for frame in frame_storage:
            if self.track_id in frame:
                frame.remove(self.track_id)
        tracks.pop(self.track_id)
        del self

    def split_track(self, frame_num: int, frame_storage: list[list]) -> 'PTrack':
        new_data: list = [None for _ in range(frame_num)]
        new_data.extend(self.frames_data[frame_num:])
        for i in range(frame_num, PTrack.total_count):
            self.frames_data[i] = None
        new_track = PTrack(new_data)
        assert len(frame_storage) != len(new_data), f"Разные размеры: {len(frame_storage)=}, {len(new_data)=}"
        for i in range(len(frame_storage)):
            if new_data[i] is not None:
                frame_storage[i].append(self.track_id)
        return new_track

    def get_status(self, first_n=5) -> str:
        most_probable = dict()
        for frame_data in self.frames_data:
            if frame_data is None:
                continue
            if frame_data["code"] not in most_probable:
                most_probable[frame_data["code"]] = 1
            else:
                most_probable[frame_data["code"]] += 1
        sorted_dict = [(k, most_probable[k]) for k in most_probable]
        sorted_dict = sorted(sorted_dict, key=lambda x: x[1], reverse=True)
        result = "\n".join(map(str, sorted_dict[:first_n]))
        return result

    def __repr__(self):
        return f"PTrack: {self.track_id}"
