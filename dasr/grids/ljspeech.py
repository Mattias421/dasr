from dora import Explorer, Launcher

@Explorer
def explorer(launcher: Launcher):
    launcher(
             [ "+augment.fastspeech2.preproces=preprocess_ljspeech"],
            task="train_augment",
             augment="fastspeech2",
             data="ljspeech",
             )
