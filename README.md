# KalyanaProject
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from collections import deque
import time
import tkinter as tk
from tkinter import messagebox

def get_graph_from_user():
    graph = {}
    num_edges = int(input("Enter number of roads (connections) in your network: "))
    print("\nPlease enter the details for each road (e.g., Delhi Agra 5 1.2):")
    for i in range(num_edges):
        print(f"\nConnection {i+1}:")
        src = input("  Source location: ").strip()
        dst = input("  Destination location: ").strip()
        dist = float(input("  Distance (in km): "))
        traffic = float(input("  Traffic factor (1.0 - 1.7): "))
        if src not in graph:
            graph[src] = {}
        graph[src][dst] = (dist, traffic)
        bidir = input("  Is this road bidirectional? (yes/no): ").strip().lower()
        if bidir == 'yes':
            if dst not in graph:
                graph[dst] = {}
            graph[dst][src] = (dist, traffic)
    return graph
def create_dataset(graph):
    data = []
    for src in graph:
        for dst in graph[src]:
            dist, traffic = graph[src][dst]
            cost = dist * traffic
            data.append({'src': src, 'dst': dst, 'distance': dist, 'traffic': traffic, 'cost': cost})
    return pd.DataFrame(data)
def train_model(df):
    le_src = LabelEncoder()
    le_dst = LabelEncoder()
    df['src_enc'] = le_src.fit_transform(df['src'])
    df['dst_enc'] = le_dst.fit_transform(df['dst'])
    X = df[['src_enc', 'dst_enc', 'distance', 'traffic']]
    y = df['cost']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("\n📊 Model Trained!")
    print(f"Mean Squared Error on test set: {mse:.3f}")
    return model, le_src, le_dst
def predict_cost(model, le_src, le_dst, src, dst, distance, traffic):
    src_enc = le_src.transform([src])[0]
    dst_enc = le_dst.transform([dst])[0]
    X_input = pd.DataFrame([[src_enc, dst_enc, distance, traffic]],
                           columns=['src_enc', 'dst_enc', 'distance', 'traffic'])
    predicted = model.predict(X_input)[0]
    return predicted
def bfs_path(graph, start, goal):
    queue = deque([[start]])
    visited = set()
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return None 
def display_path_with_icons(source, destination, graph):
    path = bfs_path(graph, source, destination)
    if not path:
        return "No path found."
    icon_path = []
    for i, node in enumerate(path):
        if i == 0:
            icon_path.append(f"📍{node}")
        elif i == len(path) - 1:
            icon_path.append(f"🏁{node}")
        else:
            icon_path.append(f"🚏{node}")
    return " 🛣 ".join(icon_path)
graph = get_graph_from_user()
df = create_dataset(graph)
print(df)
model, le_src, le_dst = train_model(df)
while True:
    a = input("Enter the Source:").strip()
    b = input("Enter the Destination:").strip()
    t = float(input("Enter the Traffic Density in range of (1.0–1.7): "))
    d = float(input("Enter the Distance between Them (in km): "))
    if a not in df['src'].unique() or b not in df['dst'].unique():
        print("❌ Source or destination not in training data.")
        continue
    print("\n🔮 Predicting cost for Given Inputs")
    predicted_cost = predict_cost(model, le_src, le_dst, a, b, d, t)
    print(f"Predicted cost: {predicted_cost:.2f}")
    print("\nPath Between Given Source and Destination:")
    if a in graph and b in graph:
        print(f"Best route from {a} to {b}:\n{display_path_with_icons(a, b, graph)}")
    else:
        print("Path is not available between given source and destination.")
    time_minutes = float(input("Enter estimated time (in minutes): "))
    time_hours = time_minutes / 60
    if time_hours > 0:
        max_speed = d / time_hours
        min_speed = (d * t) / time_hours
        print("\n🚗 Speed Analysis:")
        print(f"➡️  Maximum Speed (ignoring traffic): {max_speed:.2f} km/h")
        print(f"⬅️  Minimum Speed (with traffic factor {t}): {min_speed:.2f} km/h")
    else:
        print("❌ Invalid time entered. Cannot compute speed.")
def run_gui(model, le_src, le_dst, graph, df):
    def predict():
        src = entry_src.get().strip()
        dst = entry_dst.get().strip()
        try:
            distance = float(entry_distance.get())
            traffic = float(entry_traffic.get())
            time_minutes = float(entry_time.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers.")
            return
        if src not in df['src'].unique() or dst not in df['dst'].unique():
            messagebox.showerror("Error", "Source or Destination not found in training data.")
            return
        predicted = predict_cost(model, le_src, le_dst, src, dst, distance, traffic)
        result = f"Predicted Cost: {predicted:.2f}\n"
        route = display_path_with_icons(src, dst, graph)
        result += f"\nRoute: {route}\n"
        time_hours = time_minutes / 60
        if time_hours > 0:
            max_speed = distance / time_hours
            min_speed = (distance * traffic) / time_hours
            result += f"\n🚗 Max Speed: {max_speed:.2f} km/h\n⬅️ Min Speed (with traffic): {min_speed:.2f} km/h"
        else:
            result += "\n❌ Invalid time entered."
        output_label.config(text=result)
    root = tk.Tk()
    root.title("Route Cost Predictor")
    tk.Label(root, text="Source:").grid(row=0, column=0)
    entry_src = tk.Entry(root)
    entry_src.grid(row=0, column=1)
    tk.Label(root, text="Destination:").grid(row=1, column=0)
    entry_dst = tk.Entry(root)
    entry_dst.grid(row=1, column=1)
    tk.Label(root, text="Distance (km):").grid(row=2, column=0)
    entry_distance = tk.Entry(root)
    entry_distance.grid(row=2, column=1)
    tk.Label(root, text="Traffic (1.0 - 1.7):").grid(row=3, column=0)
    entry_traffic = tk.Entry(root)
    entry_traffic.grid(row=3, column=1)
    tk.Label(root, text="Time (minutes):").grid(row=4, column=0)
    entry_time = tk.Entry(root)
    entry_time.grid(row=4, column=1)
    tk.Button(root, text="Predict Cost", command=predict).grid(row=5, column=0, columnspan=2, pady=10)
    output_label = tk.Label(root, text="", justify="left", wraplength=300)
    output_label.grid(row=6, column=0, columnspan=2)
    root.mainloop()
run_gui(model,le_src,le_dst,graph,df)
