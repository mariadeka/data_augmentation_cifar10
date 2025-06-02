import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Ustawienia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #wszystkie obliczenia na GPU,a jak nie to na procesorze
BATCH_SIZE = 128 # tyle obrazkow w jednej iteracji uczenia
EPOCHS = 20 # tyle razy paczka przejdzie przez siec
LEARNING_RATE = 0.001

# 2. Transformacje bez augmentacji
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std =(0.247, 0.243, 0.261))  #wartości zostały policzone w osobnym skrypcie
])

# 3. CIFAR-10 – dane treningowe i testowe
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Prosta sieć CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10): #num classes = 10 <- w naszym zbiorze jest 10 klas, obrazki w zbiorze maja wymiar 32x32 pikseli
        super(SimpleCNN, self).__init__()
        self.conv_block = nn.Sequential( #to jest kontener #siec konwolucyjna, to rodzaj sieci neuronowej do przetwarzania danych, które są postaci siatki (np. jak tutaj obrazy)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.ReLU(), #max(0, wartosc), funkcja aktywacji wplywa na wyciecie umierajacych neuronow
            nn.MaxPool2d(2),  # 32x16x16 - 32 to liczba kanalow

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x8x8
            #pierwszy blok uczy sie jakichs prostszych wzorcow np krawedzi, jednolitych plam, kolorow itp.
            #drugi blok uczy sie bardziej zlozonych wzorców, przyjmuje 32 te co wyszly ostatnio i wypuszcza 4
        ) #blok konwolucyjny to zestaw cech, ktore realizuja ekstrakcję cech
        self.fc = nn.Sequential(
            nn.Flatten(), #spłaszcza do wymiary 64^2
            nn.Linear(64 * 8 * 8, 256), #y = Wx + b gdzie w to taka macierz zeby wyszlo ostatecznie tyle cech ile ma byc
            nn.ReLU(),
            nn.Linear(256, num_classes)
        ) # to ze sa jakby 2 etapy, pozwala sieci na bycie nieliniowa

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x

# 5. Funkcja treningu
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # bo domyslnie gradienty sa sumowane
        outputs = model(images) #przepuszcza obrazy przez kolejne warstwy sieci
        loss = criterion(outputs, labels) #sredni blad na wszystkich probkach
        loss.backward() #liczymy od tylu wszystkie gradienty lancuchowo
        optimizer.step() # moment nauki

        total_loss += loss.item() #dodajemy ile zle sklasyfikowal
        _, predicted = outputs.max(1) #outputs zwracca dwuelementowa krotke (wartosci_max, indeksy_max) dla kazdej probki wzdluz wymiaru 1,
        #indeksy max to etykieta o maksymalnym indeksie - czyli nasza prognoza
        correct += predicted.eq(labels).sum().item() #dodajemy te dobrze przewidziane
        total += labels.size(0) #batch_size

    accuracy = 100 * correct / total
    return total_loss / len(dataloader), accuracy

# 6. Funkcja testowania
def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return total_loss / len(dataloader), accuracy

# 7. Trening modelu
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

import matplotlib.pyplot as plt

# Listy do zapisywania wyników
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Pętla treningowa (zmodyfikuj ją)
for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_loss, test_acc = test(model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Epoka {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

# Rysowanie wykresów
plt.figure(figsize=(12, 5))

# Wykres strat
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Wykres dokładności
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')  # Zapisz wykres
plt.show()
