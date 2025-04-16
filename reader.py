import numpy as np
import cv2

class Tetris:
    width = 10
    height = 20
    board_array = np.zeros((20, 10))
    mask = None
    grey = [57,57,57]
    x=y=w=h = None
    pieces = {"i": [15,155,215], #light blue
               "o": [227, 159, 2], #yellow
               "t": [175,41,138], #purple
               "j": [227,91,2], #orange
               "l": [33,65,198], #blue
               "s": [89,177,1], #green
               "z": [215,15,55]} #red
    def __init__(self):
        for key in self.pieces:
            bgr_color = self.pieces[key]
            bgr_color = np.flip(np.array(bgr_color, dtype='uint8'))
            self.pieces[key] = bgr_color
    def queue(self, img):
        arr = []
        for key in self.pieces:
            bgr_color = self.pieces[key]
            mask = cv2.inRange(img, bgr_color, bgr_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                arr.append((key, y))
        arr = sorted(arr, key=lambda x: x[1])
        arr = [item[0] for item in arr]
        return img, arr
    def held(self, img):
        arr = []
        for key in self.pieces:
            bgr_color = self.pieces[key]
            mask = cv2.inRange(img, bgr_color, bgr_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                arr.append((key, y))
        arr = sorted(arr, key=lambda x: x[1])
        arr = [item[0] for item in arr]
        return img, arr[0] if arr else None
    def locate_board(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        max_rectangle = None

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    max_rectangle = approx
        if max_rectangle is not None:
            x, y, w, h = cv2.boundingRect(max_rectangle)
            self.x, self.y, self.w, self.h = x,y,w,h
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.imwrite("res_board.png", img)
        return img
    def board(self, img):
        flag = True
        #if self.mask is None:
        for key in self.pieces:
            bgr_color = self.pieces[key]
            mask = cv2.inRange(img, bgr_color, bgr_color)
            if flag is True:
                self.mask = mask
                flag = False
            else:
                self.mask = cv2.bitwise_or(self.mask, mask)
        result = cv2.bitwise_and(img, img, mask=self.mask)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        arr = []
        for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                arr.append((cnt, y))
        arr = sorted(arr, key=lambda x: x[1])
        arr = [item[0] for item in arr]
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        height, width, channels = img.shape
        cell_width = width/self.width
        cell_height = height/self.height
        for row in range(self.height):
            for col in range(self.width):
                x1 = int(col * cell_width)
                y1 = int(row * cell_height)
                x2 = int((col + 1) * cell_width)
                y2 = int((row + 1) * cell_height)
                cell = result[y1:y2, x1:x2]
                if np.sum(cell > 0) > (cell_width * cell_height * 0.5):
                    self.board_array[row, col] = 1
        return img, self.board_array
    def current(self, img): 
        arr = []
        for key in self.pieces:
            bgr_color = self.pieces[key]
            mask = cv2.inRange(img, bgr_color, bgr_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                arr.append((key, y))
        arr = sorted(arr, key=lambda x: x[1])
        arr = [item[0] for item in arr]
        return img, arr[0] if arr else None
    def processSS(self, img):
        if self.x is None:
            self.locate_board(img)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
        img[self.y:self.y+self.h, self.x:self.x+self.w], board_array = self.board(img[self.y:self.y+self.h, self.x:self.x+self.w])
        img[self.y:self.y+self.h, self.x+self.w:self.x+self.w+int(0.55*self.w)], queue = self.queue(img[self.y:self.y+self.h, self.x+self.w:self.x+self.w+int(0.55*self.w)])
        img[self.y:self.y+self.h, self.x-self.w:self.x], held = self.held(img[self.y:self.y+self.h, self.x-self.w:self.x])
        _, current = self.current(img[self.y:self.y+self.h, self.x:self.x+self.w])
        #cv2.imwrite("res_board.png", img)
        return img, board_array, queue, held, current