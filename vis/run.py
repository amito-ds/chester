from diamond.data_augmentation.augmentation import ImageAugmentation
from diamond.image_caption.image_caption_class import ImageDescription
from diamond.image_object_detection.image_object_class import ImageObjectDetection
from diamond.model_training.best_model import ImageModelsTraining
from diamond.post_model_analysis.post_model import ImagePostModelAnalysis
from diamond.user_classes import ImagesData, ImagesAugmentationInfo, ImageModels, ImagePostModelSpec, \
    ImageDescriptionSpec

from diamond.utils import index_labels
from vis.grayscale.grayscale import VideoToGrayScale
from vis.reversed_order.video_revereser import VideoReverser
from vis.user_classes import VideoData
from diamond import run as diamond_run
from vis.video_zoom.zoom_class import VideoZoomer


def run(video,
        image_shape=None,
        frame_per_second=None,
        plot=True,
        get_frames_description=False,
        detect_frame_objects=False,
        detect_faces=False,
        get_grayscale=False,
        get_reveresed_video=False,
        get_zoomed_video=False, zoom_factor=1.5, zoom_center=None,
        plot_sample=16
        ):
    vis_collector = {}

    video_data = VideoData(cap=video, image_shape=image_shape, frame_per_second=frame_per_second)
    vis_collector["video_data"] = video_data

    if get_grayscale:
        video_gray_scale = VideoToGrayScale(video_data)
        video_gray_scale.run(play=plot)
        vis_collector["gray scaled frames"] = video_gray_scale.grayscale_frames

    if get_reveresed_video:
        video_reverse = VideoReverser(video_data)
        video_reverse.run(play=plot)
        vis_collector["reversed video frames"] = video_reverse.reversed_frames

    if get_zoomed_video:
        video_zoomed = VideoZoomer(video_data, zoom_factor=zoom_factor, zoom_center=zoom_center)
        video_zoomed.run(play=plot)
        vis_collector["zoomed frames"] = video_zoomed.zoomed_frames

    diamond_collector = diamond_run.run(images=video_data.images,
                                        image_shape=image_shape,
                                        plot_sample=plot_sample,
                                        get_image_description=get_frames_description,
                                        get_object_detection=detect_frame_objects,
                                        detect_faces=detect_faces,
                                        is_augment_data=False, is_post_model_analysis=False, is_train_model=False,
                                        plot=plot)

    vis_collector.update(diamond_collector)

    return vis_collector
