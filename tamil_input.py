import tkinter as tk

def tamil_text_input():

    root = tk.Tk()
    root.title("Enter Tamil Complaint")

    text = tk.Text(root, height=10, width=60, font=("Nirmala UI", 14))
    text.pack()

    result = {}

    def submit():
        result["text"] = text.get("1.0", tk.END)
        root.destroy()

    btn = tk.Button(root, text="Submit", command=submit)
    btn.pack()

    root.mainloop()

    return result.get("text", "").strip()