import torch
import torchvision.models as models
import time

def test_throughput():
    device = "cuda"
    # Use ResNet50 (the standard for DLPerf)
    model = models.resnet50().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Create dummy data (Batch size 64)
    batch_size = 64
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, 1000, (batch_size,)).to(device)

    print(f"🚀 Measuring Throughput on {torch.cuda.get_device_name(0)}...")
    
    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    # Actual Test
    torch.cuda.synchronize()
    start = time.time()
    iters = 100
    for _ in range(iters):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    end = time.time()
    total_images = batch_size * iters
    images_per_sec = total_images / (end - start)
    
    print(f"📈 Result: {images_per_sec:.2f} images/sec")

if __name__ == "__main__":
    test_throughput()