from deepface import DeepFace

#Test
result = DeepFace.analyze(img_path = 'face.png', actions = ['age', 'gender', 'race', 'emotion'])