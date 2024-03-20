import os

choices = ["python_hello_world.wav", "rust_waveglow_20220118.wav", "arpabet_nonsense.wav", "capital_nonsense.wav", "goodbye.wav"]

def help():
    print("Which audio to play?")
    for count, value in enumerate(choices):
        print(f"  {count} {value}")

def play(f):
    os.system(f"ffplay -autoexit -loglevel quiet {f}")

help()

while True:

    choice = input()

    try:
        choice = int(choice)
        f = os.path.join("./audio", choices[choice])
        play(f)
    except:
        help()
        
