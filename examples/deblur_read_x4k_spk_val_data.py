import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import random

def seed_everything(seed):
    
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_ori_index(filename): 
        # filename 002/occ008.320_f2881/0001_002.png
        filename = os.path.split(filename)[1] 
        parts = os.path.splitext(filename)[0].split('_')
        if len(parts)==1:
            return int(parts[0])*4
        elif len(parts)==2:
            return int(parts[0])*4 + int(parts[1]) + 1
        else:
            raise NotImplementedError("Error dataset filename.") 

class TFI:
    def __init__(self, spikes, spike_h, spike_w, device, search_half_window=20):
        
        self.spikes = spikes
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.search_half_window = search_half_window
        self.device = device
        self.T = spikes.shape[0]
        self.T_im = self.T - 2 * search_half_window

        if self.T_im < 0:
            raise ValueError('The length of spike stream {:d} is not enough for max_search half window length {:d}'.format(self.T, self.search_half_window))

    def spikes2frame(self, key_ts):

        key_ts = key_ts + self.search_half_window 
        if (key_ts - self.search_half_window // 2 < 0) or (key_ts + self.search_half_window // 2 > self.T):
            raise ValueError('The length of spike stream {:d} is not enough for even max_search half window length {:d} // 2 = {:d} at key time stamp {:d}'.format(self.T, self.search_half_window, self.search_half_window//2, key_ts))

        formmer_index = np.zeros([self.spike_h, self.spike_w])
        latter_index = np.zeros([self.spike_h, self.spike_w])

        start_t = max(key_ts - self.search_half_window + 1, 1)
        end_t = min(key_ts + self.search_half_window, self.T)

        for ii in range(key_ts, start_t-1, -1):
            formmer_index += ii * self.spikes[ii, :, :] * (1 - np.sign(formmer_index))
            
        for ii in range(key_ts+1, end_t+1):
            latter_index += ii * self.spikes[ii, :, :] * (1 - np.sign(latter_index))

        interval = latter_index - formmer_index
        interval[interval == 0] = 2* self.search_half_window
        interval[latter_index == 0] = 2* self.search_half_window
        interval[formmer_index == 0] = 2* self.search_half_window
        interval = interval

        Image = 255 / interval
        Image = Image.astype(np.uint8)

        return Image
    

def load_vidar_dat(args, idx, frame_cnt = None):
    data_path = args.data_path[idx]
    if isinstance(data_path, str):
        array = np.fromfile(data_path, dtype=np.uint8)
    elif isinstance(data_path, (list, tuple)):
        l = []
        for name in data_path:
            a = np.fromfile(name, dtype=np.uint8)
            l.append(a)
        array = np.concatenate(l)
    else:
        raise NotImplementedError

    height = args.resolution[-2] # height
    width = args.resolution[-1] # width

    len_per_frame = height * width // 8
    calculated_framecnt = len(array) // len_per_frame
    framecnt = frame_cnt if frame_cnt is not None else calculated_framecnt
    # framecnt = min(framecnt, 300) # Limit frame count

    spikes = []

    for i in range(framecnt):
        compr_frame = array[i * len_per_frame: (i + 1) * len_per_frame]
        blist = []
        for b in range(8):
            blist.append(np.right_shift(np.bitwise_and(compr_frame, np.left_shift(1, b)), b))

        frame_ = np.stack(blist).transpose()
        spk = np.flipud(frame_.reshape((height, width), order='C'))

        spikes.append(spk)

    return np.stack(spikes)


class spkData(Dataset):
    def __init__(self, args, idx):
        
        self.h = args.resolution[0] # height
        self.w = args.resolution[1] # width 

        # Load data 
        spike_data_raw = load_vidar_dat(args, idx)

        path_blurry = args.data_path[idx].split(os.sep)
        for i in range(len(path_blurry)):
            if 'spike_2xds' in path_blurry[i]:
                break
        path_blurry[i] = path_blurry[i].replace('spike_2xds', 'blurry_33')
        path_blurry = os.path.join(*path_blurry[:-1])
        path_blurry = [os.path.join(path_blurry, d) for d in os.listdir(path_blurry) if d.endswith('.png')]
        path_blurry.sort()
        self.path_blurry = path_blurry
        
        spk_idxs = [get_ori_index(i) for i in path_blurry] 
        spk_idxs = np.asarray(spk_idxs)
        
        x = np.linspace(0, 256, spk_idxs.shape[0])
        self.slope, self.intercept = np.polyfit(x, spk_idxs/15, deg=1)

        start_frame = args.start_frame
        end_frame = args.end_frame # Use None or a large number to go to the end
        if end_frame is None or end_frame > spike_data_raw.shape[0]:
             end_frame = spike_data_raw.shape[0]

        if start_frame >= end_frame:
            raise ValueError(f"Start frame ({start_frame}) must be less than end frame ({end_frame})")

        spike_data_raw = spike_data_raw[start_frame:end_frame] # Temporal slicing

        # Check if slicing resulted in empty tensor
        if spike_data_raw.shape[0] == 0:
             raise ValueError(f"No frames selected with start={start_frame}, end={end_frame}. Max frames loaded: {spike_data_raw.shape[0]}")

        # Store final resolution (T, H, W) - T comes from the diff data
        self.T = spike_data_raw.shape[0]
        self.H = spike_data_raw.shape[1]
        self.W = spike_data_raw.shape[2]
        args.resolution = (self.T, self.H, self.W) # Update args with actual T, H, W
        print(f"Actual dataset resolution (T, H, W): {args.resolution}")

        self.spike_data = TFI(spike_data_raw, self.H, self.W, args.device)
        self.t_im = self.spike_data.T_im

    def __len__(self):
        return self.t_im

    def __getitem__(self, idx):
        spk = self.spike_data.spikes2frame(idx)
        frame_idx = int(self.slope*idx + self.intercept-1) 
        blurry_img = cv2.imread(self.path_blurry[frame_idx])
        blurry_img = cv2.resize(blurry_img, (self.W, self.H))
        return spk, blurry_img


def main(args):

    seed_everything(args.seed)
    args.data_path = [os.path.join(args.data_path, d, "spike.dat") for d in os.listdir(args.data_path)]
    # DataLoader 
    datasets = []
    for d in range(len(args.data_path)):
        datasets.append(spkData(args, d))

    name_win = "spike_viewer"
    cv2.namedWindow(name_win, cv2.WINDOW_NORMAL)

    # Concatenate datasets  
    concatenated_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(concatenated_dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)
    for i, data in tqdm(enumerate(dataloader)):
        spk, blurry_img = data
        spk = spk[0].clip(0, 255).numpy().astype(np.uint8)
        spk = np.repeat(spk[:, :, np.newaxis], 3, axis=2)
        img = np.concatenate([spk, blurry_img[0]], axis=1)
        cv2.imshow(name_win, img)
        cv2.waitKey(1)

        

if __name__ == "__main__":
   parser = argparse.ArgumentParser('Spike_Parallel_Gauss')

   # --- Data Args ---
   parser.add_argument('--data_path', default='./x4k1000fps-spk/val_/val_spike_2xds/Type2/', help="Path to .dat file")  # for the validation set

   parser.add_argument('--resolution', nargs=2, type=int, default=[256, 256], help="Target spatial resolution (H W) to crop/process")  # resolution for the validation set
   parser.add_argument('--start_frame', type=int, default=0, help="First frame index to process")
   parser.add_argument('--end_frame', type=int, default=None, help="Frame index after the last one to process (exclusive)")

   # --- System Args ---
   parser.add_argument('--device', default='cuda:0', help="Device to use (e.g., 'cuda:0', 'cpu')")
   parser.add_argument('--seed', type=int, default=42, help="Random seed")

   args = parser.parse_args()

   main(args)