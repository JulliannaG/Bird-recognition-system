import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.neighbors import KernelDensity as KD
import csv
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSC
import seaborn as sns
from shapely.geometry import Point
import contextily as ctx
from shapely.geometry import Polygon
from collections import Counter

def load_checkpoint(model, optimizer, load_path, train_path=None, val_path=None, map_location=None):
    checkpoint = torch.load(load_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_records = None
    val_records = None
    
    if train_path is not None:
        train_records = np.load(train_path)
    if val_path is not None:
        val_records = np.load(val_path)
    return model, optimizer, epoch

def create_spectrogram(path, segment_size=5):
    samples, sample_rate = librosa.load(path, sr=None)
    frame_size = 2048
    hop_size = 1024

    frames = np.array([samples[i:i + frame_size].astype(np.float64) 
                       for i in range(0, len(samples) - frame_size + 1, hop_size)
                       if len(samples[i:i + frame_size]) == frame_size])

    window = np.hanning(frame_size)
    frames *= window
    spectrum = np.abs(np.fft.rfft(frames, n=frame_size))
    power_spectrum = spectrum ** 2
    log_spectrogram = 10 * np.log10(power_spectrum + np.finfo(float).eps)

    segments = []

    for i in range(0, len(log_spectrogram), segment_size):
        segment = log_spectrogram[i:i + segment_size, :]

        if segment.shape[0] < segment_size:
            padding = np.zeros((segment_size - segment.shape[0], segment.shape[1]))
            segment = np.vstack((segment, padding))

        segments.append(segment.flatten())

    spectrogram = np.array(segments)

    return spectrogram 

class NeuralNetwork(nn.Module):

    def __init__(self, in_channels=1, in_features=5125, num_classes=306):
        super().__init__()

        self.conv_layer1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5)
        self.conv_layer2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv_layer3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv_layer4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  
        
        x = F.relu(self.conv_layer1(x))
        x = F.relu(self.conv_layer2(x))
        x = self.max_pool1(x)
        
        x = F.relu(self.conv_layer3(x))
        x = F.relu(self.conv_layer4(x))
        x = self.max_pool2(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

class Expert_A(nn.Module):
    def __init__(self, model_path):
        super(Expert_A, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = NeuralNetwork()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005) 
        self.model, optimizer, epoch = load_checkpoint(
            self.model, optimizer, model_path, 
            map_location=device
        )
        self.epoch = epoch
        
    def forward(self, x):
        return self.model(x) 

class Expert_B:
    def __init__(self, estimators):
        self.estimators = estimators 

    def predict(self, x):
        densities = {}
        for class_label, estimator in self.estimators.items():
            
            log_density = estimator.score_samples(x)  
            densities[class_label] = np.exp(log_density)

        density_array = np.array([densities[class_label] for class_label in sorted(densities.keys())])
        return torch.tensor(density_array, dtype=torch.float32).T
    
def mixture_of_experts(segments, coordinates, expert_A, expert_B):
    
    predictions = []

    for segment in segments:
        spectrogram_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # Dodanie wymiaru batch
        output_A = expert_A(spectrogram_tensor) 

        if coordinates is not None and expert_B is not None:
            weight_A = 0.7
            weight_B = 0.3
            output_B = expert_B.predict(coordinates)
            combined_output = (weight_A * output_A) + (weight_B * output_B)
        else:
            
            combined_output = output_A

        probabilities = torch.softmax(combined_output, dim=1)
        predicted_class = probabilities.argmax(dim=1).item() 
        predictions.append(predicted_class)

    class_counts = Counter(predictions)
    chosen_class, _ = class_counts.most_common(1)[0]
    final_confidence = (np.array(predictions) == chosen_class).mean()

    return chosen_class, final_confidence

def load_audio_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio files", "*.wav *.mp3 *.mp4 *.m4a *.mpga")])
    if file_path:
        audio_file_path.set(file_path)

def classify_audio():
    file_path = audio_file_path.get()
    if not file_path:
        messagebox.showerror("Błąd", "Wybierz plik audio.")
        return
    
    try:
        segments = create_spectrogram(file_path, segment_size=5)

        latitude = lat_entry.get().strip()
        longitude = lon_entry.get().strip()
        
        use_coordinates = False
        if latitude and longitude:
            try:
                lat = float(latitude)
                lon = float(longitude)
                coordinates = torch.tensor([[lat, lon]], dtype=torch.float32)
                use_coordinates = True
            except ValueError:
                messagebox.showerror("Błąd", "Podaj prawidłowe współrzędne geograficzne lub pozostaw oba pola puste.")
                return
        elif latitude or longitude:
            messagebox.showerror("Błąd", "Podaj obie współrzędne geograficzne lub pozostaw oba pola puste.")
            return

        if use_coordinates:
            predicted_class, confidence = mixture_of_experts(segments, coordinates, expert_A, expert_B)
        else:
            predicted_class, confidence = mixture_of_experts(segments, None, expert_A, None)

        bird_name = species_names.get(predicted_class, "Nieznany gatunek")
        estimator = map_estimators[predicted_class]
        plot_density_map(estimator, bird_name, confidence)

    except Exception as e:
        messagebox.showerror("Błąd", f"Wystąpił błąd: {e}")

def load_species_names(file_path):
    species_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            species_name = row['species'].strip()
            class_id = int(row['labels'].strip())
            species_dict[class_id] = species_name
    return species_dict

def plot_density_map(estimator, class_label, confidence):
    world = gpd.read_file('ne/ne_110m_admin_0_countries.shp')
    europe = world[world['SOVEREIGNT'].isin([
        'Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 
        'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 
        'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 
        'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 
        'Kosovo', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 
        'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 
        'Poland', 'Portugal', 'Romania', 'San Marino', 'Serbia', 'Slovakia', 
        'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 
        'United Kingdom'
    ])]

    # Siatka punktów wewnątrz obszaru Polski
    x_min, y_min, x_max, y_max = 12.50886, 48.24952, 24.99739, 55.46412
    x_vals = np.linspace(x_min, x_max, 200)  
    y_vals = np.linspace(y_min, y_max, 200)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    xy_samples = np.vstack([y_grid.ravel(), x_grid.ravel()]).T  

    log_density = estimator.score_samples(xy_samples)
    density = np.exp(log_density).reshape(x_grid.shape)

    # Przezroczystość dla zerowej gęstości
    density[density < 1e-3] = np.nan  


    fig, ax = plt.subplots(figsize=(8, 8))
    europe.plot(ax=ax, color='lightgreen', edgecolor='black')

    magma_custom = LSC.from_list(
        "magma_custom", plt.get_cmap("magma_r")(np.linspace(0.3, 1, 10))
    )
    
    contour = ax.contourf(x_grid, y_grid, density, cmap=magma_custom, alpha=0.8, levels=5)  
    plt.colorbar(contour, label="Gęstość")

    # Ustawienie domyślnego widoku - poziom przybliżenia
    ax.set_xlim(12.9, 25)
    ax.set_ylim(48.3, 55.5)
    ax.set_aspect('equal', 'box')

    # Możliwośc przybliżania i oddalania mapy
    def on_click(event):
        if event.button == 1:  # Zoom in
            scale_factor = 0.5
        elif event.button == 3:  # Zoom out
            scale_factor = 2
        else:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (event.xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (event.ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([event.xdata - new_width * relx, event.xdata + new_width * (1 - relx)])
        ax.set_ylim([event.ydata - new_height * rely, event.ydata + new_height * (1 - rely)])
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.title(f'Gatunek ptaka: {class_label}')
    plt.xlabel('Długość geograficzna')
    plt.ylabel('Szerokość geograficzna')
    plt.show()

species_names = load_species_names('bird_classes.csv')


with open('models/kde_estimators_linear_sil0.01.pkl', 'rb') as f:
    classification_estimators = pickle.load(f)

with open('models/kde_estimators_linear_sil0.1.pkl', 'rb') as f:
    map_estimators = pickle.load(f)

model = torch.load('models/model_checkpoint_8.pth', map_location=torch.device('cpu'), weights_only=True)
expert_B = Expert_B(classification_estimators)
expert_A = Expert_A('models/model_checkpoint_8.pth')
expert_A.eval()

root = tk.Tk()
icon_path = "bird.ico" 
root.iconbitmap(icon_path)
root.title("Klasyfikacja śpiewu ptaków")

# Pole do wczytania pliku audio
audio_file_path = tk.StringVar()
audio_button = tk.Button(root, text="Wybierz plik audio", command=load_audio_file)
audio_button.pack(pady=10)

# Pole do wyświetlania ścieżki pliku
audio_label = tk.Entry(root, textvariable=audio_file_path, width=50, state="readonly")
audio_label.pack(pady=5)

# Pola do wprowadzenia współrzędnych
lat_label = tk.Label(root, text="Szerokość geograficzna:")
lat_label.pack()
lat_entry = tk.Entry(root)
lat_entry.pack()

lon_label = tk.Label(root, text="Długość geograficzna:")
lon_label.pack()
lon_entry = tk.Entry(root)

# Przycisk do uruchomienia klasyfikacji
lon_entry.pack()
classify_button = tk.Button(root, text="Klasyfikuj", command=classify_audio)
classify_button.pack(pady=20)

root.mainloop()

