licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "reinforcement",
    srcs = ["reinforcement.py"],
    data = [
        "@minigo//cc:main",
        "@leela//:leela_0110_linux_x64",
    ] + glob([
        "puzzles/*.sgf",
    ]) + glob([
        "target-model/fifty-fifty.pb",
    ]),
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@minigo//:minigo_py",
    ],
)
