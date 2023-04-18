import cv2
from shot_detect import shot_detector


def video_marker(input_video, out_name, shots):
    # open the video file
    video = cv2.VideoCapture(input_video)

    # get the video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_name, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        shot_num = 1
        for i in shots:
            if i[0] <= frame_count <= i[1]:
                break
            shot_num += 1
        new_frame = frame.copy()
        cv2.putText(new_frame, f"Shot {shot_num}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(new_frame)

        frame_count += 1

    # release the resources
    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    o1 = shot_detector("Movie_1_new.mp4")
    o2 = shot_detector("Movie_2_new.mp4")
    o3 = shot_detector("Movie_3_new.mp4")

    video_marker("Movie_1_new.mp4", "Movie1_marked.mp4", o1)
    video_marker("Movie_2_new.mp4", "Movie2_marked.mp4", o2)
    video_marker("Movie_3_new.mp4", "Movie3_marked.mp4", o3)

