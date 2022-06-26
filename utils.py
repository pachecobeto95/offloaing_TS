from torchvision import datasets, transforms
import torch, os, sys, torch, random
import numpy as np
import config

class LoadDataset():
	def __init__(self, args, model_id):

		#Th following lines sets the initial and global hyperparameters.
		self.input_resize = args.input_resize
		self.input_dim = args.input_dim
		self.batch_size_train = args.batch_size_train
		self.batch_size_test = args.batch_size_test
		self.model_id = model_id
		self.split_ratio = args.split_ratio
		self.seed = args.seed

		# Note that we apply data augmentation in the training dataset.
		self.transformations_train = transforms.Compose([
			transforms.Resize(self.input_resize),
			transforms.CenterCrop(self.input_dim),
			#transforms.RandomChoice([transforms.ColorJitter(brightness=config.brightness)]),
			#transforms.RandomHorizontalFlip(p = config.h_flip_prob),
			#transforms.RandomRotation(config.rotation_angle),
			transforms.ToTensor(), 
			transforms.Normalize(mean = config.mean, std = config.std),
			])

		# Note that we do not apply data augmentation in the test dataset.
		self.transformations_test = transforms.Compose([
			transforms.Resize(self.input_resize),
			transforms.CenterCrop(self.input_dim)
			transforms.ToTensor(), 
			transforms.Normalize(mean=config.mean, std=config.std),
			])


	def get_indices(idx, size):
		selected_idx = random.choices(idx, k=size)
		idx = list(set(idx).difference(selected_idx))
		return selected_idx, idx

	def splitting_dataset(dataset_path):

		categories_list = os.listdir(dataset_path)

		total_nr_images = sum([len(os.listdir(os.path.join(dataset_path, category) )) for category in categories_list])
		indices = np.arange(total_nr_images)
		start = 0

		train_indices, val_indices, test_indices = [], [], []
		train_size, val_size, test_size = 25, 5, 50

		for category in categories_list:
			img_list = os.listdir(os.path.join(dataset_path, category))
			idx = np.arange(start, start+len(img_list))

			train_idx, idx = self.get_indices(idx, train_size)
			val_idx, idx = self.get_indices(idx, val_size)
			test_idx, idx = self.get_indices(idx, test_size)

			train_indices.extend(train_idx), val_indices.extend(val_idx), test_indices.extend(test_idx)
			start = len(img_list)

		return train_indices, val_indices, test_indices 


	def caltech256(self, dataset_path, idx_path):

		# This method loads the Caltech-256 dataset.

		torch.manual_seed(self.seed)
		np.random.seed(seed=self.seed)

		# This block receives the dataset path and applies the transformation data. 
		train_set = datasets.ImageFolder(dataset_path, transform=self.transformations_train)
		val_set = datasets.ImageFolder(dataset_path, transform=self.transformations_test)
		test_set = datasets.ImageFolder(dataset_path, transform=self.transformations_test)

		train_idx_path = os.path.join(idx_path, "training_idx_caltech256_id_%s.npy"%(self.model_id))
		val_idx_path = os.path.join(idx_path, "validation_idx_caltech256_id_%s.npy"%(self.model_id))
		test_idx_path = os.path.join(idx_path, "test_idx_caltech256_id_%s.npy"%(self.model_id))


		if (not os.path.exists(idx_path)):
			os.makedirs(idx_path)

		if( os.path.exists(train_idx_path) ):
			#Load the indices to always use the same indices for training, validating and testing.
			train_idx = np.load(train_idx_path)
			val_idx = np.load(val_idx_path)
			test_idx = np.load(test_idx_path)

		else:
			# This line get the indices of the samples which belong to the training dataset and test dataset. 
			train_idx, val_idx, test_idx = self.splitting_dataset(dataset_path)

			#Save the training, validation and testing indices.
			np.save(train_idx_path, train_idx)
			np.save(val_idx_path, val_idx)
			np.save(test_idx_path, test_idx)


		train_data = torch.utils.data.Subset(train_set, indices=train_idx)
		val_data = torch.utils.data.Subset(val_set, indices=val_idx)
		test_data = torch.utils.data.Subset(test_set, indices=test_idx)

		train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size_train, shuffle=True, num_workers=4)
		val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size_test, num_workers=4)
		test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size_test, num_workers=4)

		return train_loader, val_loader, val_loader
