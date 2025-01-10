import pynvim

nvim = pynvim.attach(
    "child",
    argv=[
        "nvim",
        "--embed",
        "--headless",
        "--clean",  # or '-u', 'NONE'
    ],
)

print(nvim.funcs.mode())
