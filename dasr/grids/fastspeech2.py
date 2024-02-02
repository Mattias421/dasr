from dora import Explorer

@Explorer
def explorer(launcher):

    sub = launcher.bind({'task':'train_augment',
                        'augment':'fastspeech2'})
    
    sub()