import pygame
import time

def metronome(bpm, taktart, takter):
    pygame.mixer.init()
    downbeat = pygame.mixer.Sound('Sounds/Metronome_downbeat.wav')
    upbeat = pygame.mixer.Sound('Sounds/Metronome_234.wav')

    interval = 60 / bpm
    total_beats = takter * taktart

    for i in range(total_beats):
        if i % taktart == 0:
            downbeat.play()
        else:
            upbeat.play()
        time.sleep(interval)

if __name__ == '__main__':
    bpm = int(input('Ange BPM: '))
    taktart = int(input('Hur många slag ska det vara i varje takt?: '))
    takter = int(input('Hur många takter vill du spela?: '))
    metronome(bpm,taktart, takter)

