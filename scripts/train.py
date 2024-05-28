import torch
from torch.utils.data import DataLoader
from dataset import ObjectDetectionDataset
from model import TwoStageDetector
from tqdm import tqdm

def training_loop(model, learning_rate, train_dataloader, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss_list = []
    for i in tqdm(range(n_epochs)):
        total_loss = 0
        for img_batch, gt_bboxes_batch, gt_classes_batch in train_dataloader:
            loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_list.append(total_loss)
    return loss_list

if __name__ == "__main__":
    annotation_path = "data/annotations.xml"
    image_dir = os.path.join("data", "images")
    name2idx = {'pad': -1, 'camel': 0, 'bird': 1}
    img_size = (640, 480)
    dataset = ObjectDetectionDataset(annotation_path, image_dir, img_size, name2idx)
    dataloader = DataLoader(dataset, batch_size=2)
    model = TwoStageDetector(img_size, (15, 20), 2048, 2, (2, 2))
    learning_rate = 1e-3
    n_epochs = 1000
    loss_list = training_loop(model, learning_rate, dataloader, n_epochs)
    torch.save(model.state_dict(), "model.pt")
