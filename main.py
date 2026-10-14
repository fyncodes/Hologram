import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Initialize GLUT
glutInit()
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
glutInitWindowSize(640, 480)
#glutCreateWindow(b"3D Hologram")

# Initialize Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# OpenCV Webcam Setup
cap = cv2.VideoCapture(0)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

# Pygame + OpenGL Setup
pygame.init()
screen = pygame.display.set_mode((frame_width, frame_height), DOUBLEBUF | OPENGL)
gluPerspective(45, (frame_width / frame_height), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Ask for hologram selection
print("Select a hologram:")
print("1 → Glowing Cube")
print("2 → Spinning Sphere")
print("3 → Futuristic Floating UI")
print("4 → Your Face as a Hologram")
holo_type = int(input("Enter number: "))

# Hologram properties
holo_x, holo_y = 0, 0
holo_scale = 1.0
holo_rotation = 0
prev_distance = None
prev_angle = None

def draw_cube():
    glColor3f(0.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    vertices = [
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]
    ]
    faces = [
        [0, 1, 2, 3], [1, 5, 6, 2], [5, 4, 7, 6], [4, 0, 3, 7], [3, 2, 6, 7], [4, 5, 1, 0]
    ]
    for face in faces:
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_sphere():
    glColor3f(1.0, 0.5, 0.0)
    quadric = gluNewQuadric()
    gluSphere(quadric, 1, 30, 30)  # Radius = 1, 30 segments

def draw_floating_ui():
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINE_LOOP)
    for angle in range(0, 360, 10):
        rad = np.radians(angle)
        glVertex3f(np.cos(rad), np.sin(rad), 0)
    glEnd()

def draw_face_hologram():
    glColor3f(0.5, 0.5, 1.0)
    glBegin(GL_LINES)
    for x in np.linspace(-1, 1, 10):
        for y in np.linspace(-1, 1, 10):
            glVertex3f(x, y, np.sin(x * y * 10) * 0.5)
    glEnd()

def draw_hologram():
    if holo_type == 1:
        draw_cube()
    elif holo_type == 2:
        draw_sphere()
    elif holo_type == 3:
        draw_floating_ui()
    elif holo_type == 4:
        draw_face_hologram()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    palm_x, palm_y = None, None
    palm_width = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            handedness = result.multi_handedness[result.multi_hand_landmarks.index(hand_landmarks)].classification[0].label

            wrist = hand_landmarks.landmark[0]
            middle_mcp = hand_landmarks.landmark[9]

            palm_x = (wrist.x + middle_mcp.x) / 2
            palm_y = (wrist.y + middle_mcp.y) / 2

            index_x = hand_landmarks.landmark[8].x
            pinky_x = hand_landmarks.landmark[20].x
            palm_width = abs(index_x - pinky_x)

    # Move hologram based on palm position
    if palm_x is not None and palm_y is not None:
        holo_x = (palm_x - 0.5) * 2
        holo_y = -(palm_y - 0.5) * 2

    # Scale hologram based on palm width
    if palm_width is not None:
        holo_scale = max(0.5, min(2.5, palm_width * 10))

    # OpenGL Rendering
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPushMatrix()
    glTranslatef(holo_x, holo_y, 0)
    glScalef(holo_scale, holo_scale, holo_scale)
    glRotatef(holo_rotation, 0, 1, 0)  # Rotate on Y-axis
    draw_hologram()
    glPopMatrix()
    pygame.display.flip()

    # Convert OpenGL to Image & Overlay on Facecam
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, frame_width, frame_height, GL_RGB, GL_UNSIGNED_BYTE)
    opengl_frame = np.frombuffer(pixels, dtype=np.uint8).reshape((frame_height, frame_width, 3))
    opengl_frame = cv2.flip(opengl_frame, 0)

    blended = cv2.addWeighted(frame, 0.6, opengl_frame, 0.4, 0)
    cv2.imshow("3D Hologram Facecam", blended)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
