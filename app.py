import os
from os import listdir
import gradio as gr
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

from dataset import GlyphData
from model import Glyphnet


dist = { 0: "A55", 1: "Aa15", 2: "Aa26", 3: "Aa27", 4: "Aa28", 5: "D1", 6: "D10", 7: "D156", 8: "D19", 9: "D2", 10: "D21", 11: "D28", 12: "D34", 13: "D35", 14: "D36", 15: "D39", 16: "D4", 17: "D46", 18: "D52", 19: "D53", 20: "D54", 21: "D56", 22: "D58", 23: "D60", 24: "D62", 25: "E1", 26: "E17", 27: "E23", 28: "E34", 29: "E9", 30: "F12", 31: "F13", 32: "F16", 33: "F18", 34: "F21", 35: "F22", 36: "F23", 37: "F26", 38: "F29", 39: "F30", 40: "F31", 41: "F32", 42: "F34", 43: "F35", 44: "F4", 45: "F40", 46: "F9", 47: "G1", 48: "G10", 49: "G14", 50: "G17", 51: "G21", 52: "G25", 53: "G26", 54: "G29", 55: "G35", 56: "G36", 57: "G37", 58: "G39", 59: "G4", 60: "G40", 61: "G43", 62: "G5", 63: "G50", 64: "G7", 65: "H6", 66: "I10", 67: "I5", 68: "I9", 69: "L1", 70: "M1", 71: "M12", 72: "M16", 73: "M17", 74: "M18", 75: "M195", 76: "M20", 77: "M23", 78: "M26", 79: "M29", 80: "M3", 81: "M4", 82: "M40", 83: "M41", 84: "M42", 85: "M44", 86: "M8", 87: "N1", 88: "N14", 89: "N16", 90: "N17", 91: "N18", 92: "N19", 93: "N2", 94: "N24", 95: "N25", 96: "N26", 97: "N29", 98: "N30", 99: "N31", 100: "N35", 101: "N36", 102: "N37", 103: "N41", 104: "N5", 105: "O1", 106: "O11", 107: "O28", 108: "O29", 109: "O31", 110: "O34", 111: "O4", 112: "O49", 113: "O50", 114: "O51", 115: "P1", 116: "P13", 117: "P6", 118: "P8", 119: "P98", 120: "Q1", 121: "Q3", 122: "Q7", 123: "R4", 124: "R8", 125: "S24", 126: "S28", 127: "S29", 128: "S34", 129: "S42", 130: "T14", 131: "T20", 132: "T21", 133: "T22", 134: "T28", 135: "T30", 136: "U1", 137: "U15", 138: "U28", 139: "U33", 140: "U35", 141: "U7", 142: "V13", 143: "V16", 144: "V22", 145: "V24", 146: "V25", 147: "V28", 148: "V30", 149: "V31", 150: "V4", 151: "V6", 152: "V7", 153: "W11", 154: "W14", 155: "W15", 156: "W18", 157: "W19", 158: "W22", 159: "W24", 160: "W25", 161: "X1", 162: "X6", 163: "X8", 164: "Y1", 165: "Y2", 166: "Y3", 167: "Y5", 168: "Z1", 169: "Z11", 170: "Z7" }



model = Glyphnet(num_classes=171,
                    first_conv_out=64,
                    last_sconv_out=512,
                    sconv_seq_outs= [128, 128, 256, 256],
                    dropout_rate=0.1
                    )


model_path = './checkpoint/best_checkpoint.pth'
model.load_state_dict(torch.load(model_path))

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model.to(torch.device(device))

def predict(input,model=model,device=device):
    """ Testing an already trained model using the provided data from `test_loader` """
    model.eval()
    input= input["composite"]
    input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0).float()
    
    print(input)
    input = input.to(device)
    with torch.no_grad():
        output = model(input)
        print(output)
        print(torch.max(output, 1))
        outputs = output.argmax(1).cpu().numpy()[0]
        print(outputs)
        pred = dist[outputs]
        return pred
            



interface = gr.Interface(fn=predict, inputs=gr.ImageEditor(type="numpy",image_mode="L",transforms=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                        transforms.ToTensor()]),crop_size=(100,100)),outputs="text",title="Egyptian-Classification-master")

interface.launch()