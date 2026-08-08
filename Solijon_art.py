import numpy as np
import cv2

# CANVAS
canvas = np.zeros((140, 250, 3), dtype = "uint8")
CANVAS_Y, CANVAS_X = canvas.shape[:2]

# COLORS
red = (0, 0, 255)
blue = (255, 0, 0)
white = (245, 245, 245)
black = (0, 0, 0)
dark_gray = (33, 33, 33)
gray = (45, 45, 45)
light_gray = (160, 160, 160)
orange = (51, 153, 255)
dark_orange = (0, 76, 153)
night = (20, 10, 10)
yellow = (0, 255, 255)
silver = (192, 192, 192)
dark_silver = (152, 152, 152)

def display_drawing():
    # BACKGROUND
    j = 0
    for i in range(0, 7):
        cv2.rectangle(canvas, (0, 0+j), (CANVAS_X, 15*(i+1)), ((25 + (5*i)), (0 + (0*i)), (5 + (4*i))), -1)
        j += 15

    # RAILINGS
    cv2.rectangle(canvas, (0, 74), (CANVAS_X, 81), silver, -1)
    cv2.rectangle(canvas, (0, 77), (CANVAS_X, 78), dark_silver, -1)

    k = 0
    for i in range(1, 10):
        cv2.rectangle(canvas, (25 + k, 79), (30 + k, 103), silver, -1)
        k += 60

    # FRONT SIDE
    front_arr = np.array([[134, 45], [135, 45], [136, 46], [137, 46], [138, 47], [139, 47], [160, 63], [162, 64], [163, 64], [164, 65], [173, 65],
                          [174, 66], [181, 66], [182, 67], [186, 67], [187, 68], [191, 68], [192, 69], [196, 69], [197, 70], [201, 70], [202, 71],
                          [206, 71], [207, 72], [208, 72], 
                          [209, 71], [210, 70], [211, 69], [212, 69], [213, 68], [214, 68], [215, 69], [214, 70], [214, 74], [134, 74]])
    front_side = front_arr.reshape((-1,1,2))
    cv2.fillPoly(canvas, [front_side], gray)
    cv2.polylines(canvas, [front_side], isClosed=False, color=black)

    # CAR BACKGROUND
    car_arr = np.array([[94, 43], [126, 43], [127, 44], [132, 44], [133, 45], [157, 66], [158, 66], [179, 68], [214, 75], [215, 76], [215, 78], 
                        [216, 79], [217, 80], [218, 81], [218, 86], [217, 87], [216, 88], [215, 88], [215, 91], [216, 92], [216, 94], #right side

                        [215, 95], [190, 95], [190, 80], [162, 80], [162, 96], [91, 96], [91, 81], [63, 81], [63, 91], [39, 91], #bottom part

                        [38, 90], [37, 89], [36, 89], [35, 88], [34, 87], [33, 86], [33, 82], [34, 81], [36, 81], [37, 80], [36, 79], [39, 63],
                        [40, 62], [46, 62], [47, 62], [73, 50], [72, 50], [91, 44], [89, 44], [93, 44]]) #left side
    carOutline = car_arr.reshape((-1,1,2))
    cv2.fillPoly(canvas, [carOutline], white)

    # 1st window
    windowOne = np.array([[55, 66], [80, 66], [97, 67], [98, 66], [101, 47], [94, 47], [96, 47], [87, 49], [55, 63]])
    w1_pts = windowOne.reshape((-1,1,2))
    cv2.fillPoly(canvas, [w1_pts], gray)
    cv2.polylines(canvas, [w1_pts], isClosed=True, color=black, thickness=1)

    # 2nd window
    windowTwo = np.array([[104, 66], [125, 66], [153, 67], [153, 66], [132, 48], [131, 47], [107, 47], [106, 48]])
    w2_pts = windowTwo.reshape((-1,1,2))
    cv2.fillPoly(canvas, [w2_pts], gray)
    cv2.polylines(canvas, [w2_pts], isClosed=True, color=black, thickness=1)

    # BODY PAINT
    cv2.ellipse(canvas, (77, 91), (16, 15), 0, 0, -180, gray, cv2.FILLED)
    cv2.ellipse(canvas, (176, 91), (16, 15), 0, 0, -180, gray, cv2.FILLED)

    cv2.line(canvas, (37, 79), (215, 79), gray)
    cv2.line(canvas, (38, 80), (216, 80), gray)
    cv2.line(canvas, (37, 81), (217, 81), gray)

    cv2.line(canvas, (34, 82), (68, 82), white)
    cv2.line(canvas, (86, 82), (167, 82), white)
    cv2.line(canvas, (185, 82), (217, 82), white)

    border_arr = np.array([[36, 78], [68, 78], [69, 77], [70, 76], [73, 76], [74, 75], [80, 75], [81, 76], [84, 76], [85, 77], [86, 78], [167, 78],
                           [168, 77], [169, 76], [172, 76], [173, 75], [179, 75], [180, 76], [183, 76], [184, 77], [185, 78], [215, 78]])
    paint_border = border_arr.reshape((-1,1,2))
    cv2.polylines(canvas, [paint_border], isClosed=False, color=black)

    body1_arr = np.array([[63, 91], [39, 91], [38, 90], [37, 89], [36, 89], [35, 88], [34, 87], [33, 86], [33, 83], [63, 83]])
    body1_paint = body1_arr.reshape((-1,1,2))
    cv2.fillPoly(canvas, [body1_paint], gray)

    cv2.rectangle(canvas, (91, 83), (162, 96), gray, -1)

    body2_arr = np.array([[215, 95], [190, 95], [190, 83], [218, 83], [218, 86], [217, 87], [216, 88], [215, 88], [215, 91], [216, 92], [216, 94]])
    body2_paint = body2_arr.reshape((-1,1,2))
    cv2.fillPoly(canvas, [body2_paint], gray)

    # CAR OUTLINE
    # (right side)
    cv2.line(canvas, (94, 43), (126, 43), black)
    cv2.line(canvas, (127, 44), (132, 44), black)
    cv2.line(canvas, (133, 45), (157, 66), black)
    cv2.line(canvas, (158, 66), (179, 68), black)
    cv2.line(canvas, (179, 68), (214, 75), black)
    cv2.line(canvas, (215, 76), (215, 78), black)
    canvas[79, 216], canvas[80, 217] = black, black
    cv2.line(canvas, (218, 81), (218, 86), black)
    canvas[87, 217], canvas[88, 216] = black, black
    cv2.line(canvas, (215, 88), (215, 91), black)
    cv2.line(canvas, (216, 92), (216, 94), black)

    # (left side)
    cv2.line(canvas, (93, 44), (89, 44), black)
    cv2.line(canvas, (72, 50), (91, 44), black)
    cv2.line(canvas, (73, 50), (47, 62), black)
    cv2.line(canvas, (46, 62), (40, 62), black)
    cv2.line(canvas, (39, 63), (36, 79), black)
    canvas[80, 37] = black
    cv2.line(canvas, (36, 81), (34, 81), black)
    cv2.line(canvas, (33, 82), (33, 86), black)
    points = [(87, 34), (88, 35), (89, 36), (89, 37), (90, 38)]
    for y, x in points:
        canvas[y, x] = black
    cv2.line(canvas, (39, 91), (63, 91), black)

    # (door line)
    door1_arr = np.array([[106, 44], [105, 45], [105, 47], [104, 48], [104, 52], [103, 53], [103, 61], [102, 62], [102, 67], [101, 68], [101, 70],
                         [100, 71], [100, 77]])
    door1Outline = door1_arr.reshape((-1,1,2))
    cv2.polylines(canvas, [door1Outline], isClosed=False, color=light_gray)

    door2_arr = np.array([[100, 83], [100, 87], [101, 88], [101, 90], [102, 91], [102, 92], [103, 93], [156, 93], [157, 92], [158, 91], [158, 83]])
    door2Outline = door2_arr.reshape((-1,1,2))
    cv2.polylines(canvas, [door2Outline], isClosed=False, color=black)

    door3_arr = np.array([[154, 67], [156, 67], [157, 68], [158, 70], [158, 77]])
    door3Outline = door3_arr.reshape((-1,1,2))
    cv2.polylines(canvas, [door3Outline], isClosed=False, color=light_gray)

    canvas[81, 100], canvas[80, 100], canvas[79, 100], canvas[81, 158], canvas[80, 158], canvas[79, 158] = black, black, black, black, black, black
    canvas[82, 100], canvas[82, 158] = light_gray, light_gray

    # (bottom)
    cv2.line(canvas, (91, 96), (162, 96), black)
    cv2.line(canvas, (190, 95), (215, 95), black)
    cv2.line(canvas, (91, 95), (91, 92), black)
    cv2.line(canvas, (162, 95), (162, 92), black)
    cv2.line(canvas, (190, 95), (190, 92), black)

    cv2.ellipse(canvas, (77, 91), (13, 11), 0, 0, -180, night, cv2.FILLED)
    cv2.ellipse(canvas, (77, 91), (13, 11), 0, 0, -180, black, 1)
    cv2.ellipse(canvas, (176, 91), (13, 11), 0, 0, -180, night, cv2.FILLED)
    cv2.ellipse(canvas, (176, 91), (13, 11), 0, 0, -180, black, 1)

    # WHEELS
    cv2.circle(canvas, (77, 92), 12, black, -1)
    cv2.circle(canvas, (77, 92), 9, dark_gray, -1)
    cv2.circle(canvas, (77, 92), 7, black)
    cv2.circle(canvas, (77, 92), 3, light_gray, -1)

    cv2.line(canvas, (77, 85), (77, 99), black)
    cv2.line(canvas, (70, 92), (84, 92), black)
    cv2.line(canvas, (73, 88), (81, 96), black)
    cv2.line(canvas, (81, 88), (73, 96), black)


    cv2.circle(canvas, (176, 92), 12, black, -1)
    cv2.circle(canvas, (176, 92), 9, dark_gray, -1)
    cv2.circle(canvas, (176, 92), 7, black)
    cv2.circle(canvas, (176, 92), 3, light_gray, -1)

    cv2.line(canvas, (176, 85), (176, 99), black)
    cv2.line(canvas, (169, 92), (183, 92), black)
    cv2.line(canvas, (172, 88), (180, 96), black)
    cv2.line(canvas, (180, 88), (172, 96), black)

    # HEADLIGHT
    canvas[77, 214] = dark_orange
    cv2.line(canvas, (213, 77), (210, 77), orange)
    canvas[77, 209] = dark_orange
    cv2.line(canvas, (214, 76), (209, 76), black)
    canvas[77, 208] = black
    canvas[75, 212], canvas[75, 213] = white, white

    # TAILLIGHT
    tail_arr = np.array([[37, 76], [41, 76], [43, 70], [38, 70], [37, 71], [37, 75]])
    tailOutline = tail_arr.reshape((-1,1,2))
    cv2.fillPoly(canvas, [tailOutline], red)
    cv2.polylines(canvas, [tailOutline], isClosed=False, color=black)

    # FUEL TANK
    cv2.rectangle(canvas, (53, 73), (59, 79), black)

    # DOOR HANDLE
    cv2.rectangle(canvas, (105, 69), (111, 70), dark_gray, -1)
    canvas[72, 105] = gray

    # ROAD
    road_arr = np.array([[0, 104], [CANVAS_X, 103], [CANVAS_X, CANVAS_Y], [0, CANVAS_Y]])
    road_shape = road_arr.reshape((-1,1,2))
    cv2.fillPoly(canvas, [road_shape], (15, 15, 15))
    cv2.rectangle(canvas, (30, 104), (60, 105), yellow)
    cv2.rectangle(canvas, (90, 104), (120, 105), yellow)
    cv2.rectangle(canvas, (150, 103), (180, 104), yellow)
    cv2.rectangle(canvas, (210, 103), (240, 104), yellow)

    # TEXT
    # cv2.putText(canvas, 'FUJIWARA', (117, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.25, black, thickness=1, lineType=cv2.LINE_8, bottomLeftOrigin=False)
    
# STARS
def generate_stars():
    num_stars = 20
    for i in range(num_stars):
        x = np.random.randint(5, CANVAS_X)
        y = np.random.randint(5, 41)
        canvas[y, x] = white

# LOAD DRAWING
while True:
    display_drawing()
    generate_stars()
    scale = 5
    resized = cv2.resize(canvas, (int(CANVAS_X * scale), int(CANVAS_Y * scale)), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Toyota AE86", resized)
    if cv2.waitKey(1500) & 0xFF == 27: # ESC key
        break

