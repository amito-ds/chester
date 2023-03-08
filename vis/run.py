from diamond.data_augmentation.augmentation import ImageAugmentation
from diamond.image_caption.image_caption_class import ImageDescription
from diamond.image_object_detection.image_object_class import ImageObjectDetection
from diamond.model_training.best_model import ImageModelsTraining
from diamond.post_model_analysis.post_model import ImagePostModelAnalysis
from diamond.user_classes import ImagesData, ImagesAugmentationInfo, ImageModels, ImagePostModelSpec, \
    ImageDescriptionSpec

from diamond.utils import index_labels
from vis.user_classes import VideoData
from diamond import run as diamond_run


def run(video,
        image_shape=None,
        frame_per_second=None,
        plot=True,
        get_frames_description=False,
        detect_frame_objects=False,
        plot_sample=16
        ):
    # Tell a story
    # story = """Welcome to MadCat, the comprehensive machine learning and data analysis solution!
    # \nThis module is designed to streamline the entire process of video tasks,
    # \nfrom start to finish.
    # \nTo learn more about MadCat, visit https://github.com/amito-ds/chester.\n"""
    # print(story)
    vis_collector = {}

    video_data = VideoData(cap=video, image_shape=image_shape, frame_per_second=frame_per_second)
    vis_collector["video_data"] = video_data
    # plot
    # if plot:
    #     print("Video Sample Plot")
    #     video_data.plot_video_images(plot_sample=plot_sample)

    diamond_collector = diamond_run.run(images=video_data.images,
                                        image_shape=image_shape,
                                        get_image_description=get_frames_description,
                                        get_object_detection=detect_frame_objects,
                                        is_augment_data=False, is_post_model_analysis=False, is_train_model=False,
                                        plot=plot)

    vis_collector.update(diamond_collector)

    return vis_collector
