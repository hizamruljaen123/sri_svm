import tkinter as tk
import random

# Inisialisasi jendela dan canvas
root = tk.Tk()
root.title("Game Snake")
canvas = tk.Canvas(root, width=400, height=400, bg="black")
canvas.pack()
canvas.focus_set()  # Agar canvas bisa menerima input tombol

# Label untuk skor
score_label = tk.Label(root, text="Score: 0")
score_label.pack()

# Variabel awal permainan
snake_positions = [(100, 100)]  # Posisi awal ular
snake_rects = [canvas.create_rectangle(100, 100, 120, 120, fill="green")]  # Segmen ular
direction = "right"  # Arah awal

# Fungsi untuk menentukan posisi acak makanan
def random_position():
    while True:
        x = random.randint(0, 19) * 20  # Koordinat dalam kelipatan 20 agar sesuai grid
        y = random.randint(0, 19) * 20
        if (x, y) not in snake_positions:  # Pastikan tidak di posisi ular
            return (x, y)

# Inisialisasi makanan
food_pos = random_position()
food_rect = canvas.create_rectangle(food_pos[0], food_pos[1], 
                                   food_pos[0] + 20, food_pos[1] + 20, fill="red")

score = 0
game_running = True

# Fungsi untuk menangani input tombol
def on_key_press(event):
    global direction
    new_direction = event.keysym.lower()  # Ubah ke huruf kecil: left, right, up, down
    opposites = {"left": "right", "right": "left", "up": "down", "down": "up"}
    # Hanya izinkan perubahan arah yang tidak berlawanan
    if new_direction in ["left", "right", "up", "down"] and new_direction != opposites[direction]:
        direction = new_direction

# Fungsi untuk memperbarui permainan
def update():
    global snake_positions, snake_rects, food_pos, food_rect, score, game_running
    if not game_running:
        return

    # Tentukan posisi kepala baru berdasarkan arah
    head_x, head_y = snake_positions[0]
    if direction == "right":
        new_head = (head_x + 20, head_y)
    elif direction == "left":
        new_head = (head_x - 20, head_y)
    elif direction == "up":
        new_head = (head_x, head_y - 20)
    elif direction == "down":
        new_head = (head_x, head_y + 20)

    # Cek tabrakan dengan dinding atau tubuh sendiri
    if (new_head[0] < 0 or new_head[0] >= 400 or 
        new_head[1] < 0 or new_head[1] >= 400 or 
        new_head in snake_positions):
        game_running = False
        canvas.create_text(200, 200, text="Game Over", fill="white", font=("Arial", 24))
        return

    # Tambahkan kepala baru
    snake_positions.insert(0, new_head)
    new_rect = canvas.create_rectangle(new_head[0], new_head[1], 
                                       new_head[0] + 20, new_head[1] + 20, fill="green")
    snake_rects.insert(0, new_rect)

    # Cek jika ular memakan makanan
    if new_head == food_pos:
        score += 1
        score_label.config(text="Score: {}".format(score))
        canvas.delete(food_rect)
        food_pos = random_position()
        food_rect = canvas.create_rectangle(food_pos[0], food_pos[1], 
                                           food_pos[0] + 20, food_pos[1] + 20, fill="red")
    else:
        # Hapus ekor jika tidak makan
        canvas.delete(snake_rects.pop())
        snake_positions.pop()

    # Jadwalkan pembaruan berikutnya (kecepatan permainan)
    root.after(100, update)

# Bind tombol ke fungsi on_key_press
root.bind("<Key>", on_key_press)

# Mulai permainan
update()
root.mainloop()