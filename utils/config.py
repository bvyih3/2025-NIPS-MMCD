# ----------------------running settings-------------------------- #
train_set   = 'train'   # 'train' or 'train+val'
loss_type   = 'ce_margin'
in_memory   = False     # load all the image feature in memory

# ----------------------running settings-------------------------- #
entropy = 4.5
scale = 16
alpha = 0.5
temp = 0.2
use_cos = True
sc_epoch = 30
bias_inject = True
learnable_margins = True
randomization = True
supcon = True
dataset = 'vqacp-v2'

# ----------------------preprocess image config------------------ #
num_fixed_boxes         = 36        # max number of object proposals per image
output_features         = 2048      # number of features in each object proposal

main_path = None
qa_path = None
bottom_up_path = None
glove_path = None
ids_path = None
image_path = None
resized_images_path = None
rcnn_path = None
cache_root = None
dict_path = None
glove_embed_path = None
min_occurence = 0
max_question_len = 23
trainval_num_images = 0
test_num_images = 0

def update_paths(dataset):
    global main_path, qa_path, bottom_up_path, glove_path, trainval_num_images, test_num_images
    global ids_path, image_path, resized_images_path, rcnn_path, cache_root, dict_path, glove_embed_path, max_question_len

    main_path = f'your_path/{dataset}'
    qa_path = main_path
    bottom_up_path = f'your_path/{dataset}/detection_features/'
    glove_path = f'your_path/glove.6B.300d.txt'

    ids_path = f'your_path/{dataset}'
    image_path = f'your_path/{dataset}/image'
    resized_images_path = f'your_path/{dataset}/resized_images'
    if dataset == 'vqa-v2' or dataset == 'vqacp-v2' or dataset == 'vqacp-v1' or dataset == 'vqace':
        resized_images_path = '/data/Datasets/coco/{}2014/COCO_{}2014_{}.jpg'

    rcnn_path = f'your_path/{dataset}/rcnn/'
    cache_root = f'your_path/{dataset}'
    dict_path = f'{qa_path}/dictionary.json'
    glove_embed_path = f'{main_path}/glove6b_init.npy'

    if dataset == 'vqacp-v1':
        trainval_num_images     = 118442
        test_num_images         = 87400
    elif dataset == 'vqacp-v2':
        trainval_num_images     = 120932
        test_num_images         = 98226
    elif dataset == 'gqaood':
        trainval_num_images     = 72140
        test_num_images         = 388
    elif dataset == 'vqa-v2':
        trainval_num_images     = 123287 # 82783 
        test_num_images         = 40504
