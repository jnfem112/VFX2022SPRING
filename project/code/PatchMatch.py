import argparse
import os
from tkinter import *
from tkinter import filedialog as fd
from tkinter.messagebox import askyesno , showerror
from tkinter.simpledialog import askstring
from tkinter.colorchooser import askcolor
from PIL import Image , ImageTk
import cv2
import numpy as np
from inpainter import inpainter

class GUI:
	def __init__(self):
		self.root = Tk()
		self.root.title('Patch Match')
		self.root.resizable(False , False)
		self.root.protocol('WM_DELETE_WINDOW' , self.on_closing)

		self.open_button = Button(self.root , text = 'Open' , command = self.open_file)
		self.open_button.grid(row = 0 , column = 0)

		self.pen_button = Button(self.root , text = 'Pen' , command = self.click_pen_button)
		self.pen_button.grid(row = 0 , column = 1)

		self.line_button = Button(self.root , text = 'Line' , command = self.click_line_button)
		self.line_button.grid(row = 0 , column = 2)

		self.width_button = Scale(self.root , from_ = 1 , to = 50 , orient = HORIZONTAL)
		self.width_button.grid(row = 0 , column = 3)

		self.color_button = Button(self.root , text = 'Color' , command = self.choose_color)
		self.color_button.grid(row = 0 , column = 4)

		self.undo_button = Button(self.root , text = 'Undo' , command = self.undo)
		self.undo_button.grid(row = 0 , column = 5)

		self.inpaint_button = Button(self.root , text = 'Inpaint' , command = self.inpaint)
		self.inpaint_button.grid(row = 0 , column = 6)

		self.shuffle_button = Button(self.root , text = 'Shuffle' , command = self.click_shuffle_button)
		self.shuffle_button.grid(row = 0 , column = 7)

		self.save_button = Button(self.root , text = 'Save' , command = self.save_image)
		self.save_button.grid(row = 0 , column = 8)

		self.canvas = Canvas(self.root , width = 750 , height = 500 , bg = 'white' , bd = 0 , highlightthickness = 0)
		self.canvas.grid(row = 1 , columnspan = 9)
		self.canvas.bind('<B1-Motion>' , self.start_paint)
		self.canvas.bind('<ButtonRelease-1>' , self.stop_paint)

		self.active_button = self.pen_button
		self.active_button.config(relief = SUNKEN)
		self.color = '#fefb00'

		self.has_input = False
		self.root.mainloop()

	def open_file(self):
		if self.has_input:
			answer = askyesno(title = 'Open Image' , message = 'Are you sure that you want to open a new image? Your changes will be lost if you don\'t save them.')
		if not self.has_input or answer:
			image_name = fd.askopenfilename(title = 'Open Image' , initialdir = '.' , filetypes = [('Image' , 'jpg') , ('Image' , 'jpeg') , ('Image' , 'png')])
			if image_name:
				self.has_input = True
				self.canvas.delete('all')
				image = ImageTk.PhotoImage(Image.open(image_name))
				self.canvas.config(width = image.width() , height = image.height())
				self.canvas.create_image(0 , 0 , anchor = NW , image = image)
				self.reference = [image]
				self.image_array = cv2.imread(image_name , cv2.IMREAD_COLOR)

				self.previous_x = None
				self.previous_y = None
				self.action_history = []
				self.image_array_history = []
				self.inpaint_history = [self.image_array.copy()]

	def click_pen_button(self):
		self.activate_button(self.pen_button)

	def click_line_button(self):
		self.activate_button(self.line_button)

	def click_shuffle_button(self):
		self.activate_button(self.shuffle_button)
		self.mask = np.any(self.image_array != self.inpaint_history[-1] , axis = 2)
		image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.image_array , cv2.COLOR_BGR2RGB)))
		image_id = self.canvas.create_image(0 , 0 , anchor = NW , image = image)
		self.reference.append(image)
		self.action_history.append({'type' : 'Shuffle' , 'image_id' : image_id})
		self.image_array_history.append(self.image_array.copy())

	def activate_button(self , button):
		self.active_button.config(relief = RAISED)
		self.active_button = button
		self.active_button.config(relief = SUNKEN)

	def choose_color(self):
		_ , self.color = askcolor(color = self.color)

	def Hex_to_BGR(self , color):
		return (int(color[5 : 7] , 16) , int(color[3 : 5] , 16) , int(color[1 : 3] , 16))

	def start_paint(self , event):
		if self.has_input:
			if self.active_button.cget('text') == 'Pen':
				if self.previous_x is None or self.previous_y is None:
					self.action_history.append({'type' : 'Pen' , 'line_id' : []})
					self.image_array_history.append(self.image_array.copy())
				else:
					line_id = self.canvas.create_line(self.previous_x , self.previous_y , event.x , event.y , width = self.width_button.get() , capstyle = ROUND , fill = self.color)
					self.action_history[-1]['line_id'].append(line_id)
					cv2.line(self.image_array , (self.previous_x , self.previous_y) , (event.x , event.y) , self.Hex_to_BGR(self.color) , self.width_button.get())
				self.previous_x = event.x
				self.previous_y = event.y
			elif self.active_button.cget('text') == 'Line':
				if self.previous_x is  None or self.previous_y is None:
					self.previous_x = event.x
					self.previous_y = event.y
					self.action_history.append({'type' : 'Line' , 'line_id' : None})
					self.image_array_history.append(self.image_array.copy())
				else:
					if self.action_history[-1]['line_id'] is not None:
						self.canvas.delete(self.action_history[-1]['line_id'])
					self.action_history[-1]['line_id'] = self.canvas.create_line(self.previous_x , self.previous_y , event.x , event.y , width = self.width_button.get() , capstyle = ROUND , fill = self.color)
			elif self.active_button.cget('text') == 'Shuffle':
				if self.previous_x is  None or self.previous_y is None:
					self.previous_x = event.x
					self.previous_y = event.y
				else:
					if np.any(self.mask[self.previous_y][self.previous_x]):
						translation_matrix = np.float32([[1 , 0 , event.x - self.previous_x] , [0 , 1 , event.y - self.previous_y]])
						shifted_image = cv2.warpAffine(self.inpaint_history[-1] , translation_matrix , (self.inpaint_history[-1].shape[1] , self.inpaint_history[-1].shape[0]))
						shifted_mask = cv2.warpAffine(self.mask.astype(np.uint8) , translation_matrix , (self.mask.shape[1] , self.mask.shape[0]))
						shifted_mask = np.repeat(np.expand_dims(shifted_mask , axis = 2) , 3 , axis = 2)
						original_image = self.inpaint_history[-1].copy()
						original_image[self.mask] = 0
						shffule_image = (1 - shifted_mask) * original_image + shifted_mask * shifted_image
						shffule_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(shffule_image , cv2.COLOR_BGR2RGB)))
						self.canvas.itemconfig(self.action_history[-1]['image_id'] , image = shffule_image)
						self.reference[-1] = shffule_image

	def stop_paint(self , event):
		if self.has_input:
			if self.active_button.cget('text') == 'Line' and self.previous_x is not None and self.previous_y is not None:
				cv2.line(self.image_array , (self.previous_x , self.previous_y) , (event.x , event.y) , self.Hex_to_BGR(self.color) , self.width_button.get())
			elif self.active_button.cget('text') == 'Shuffle':
				self.shuffle(event.x , event.y)
				self.activate_button(self.pen_button)
			self.previous_x = None
			self.previous_y = None

	def undo(self):
		if self.has_input:
			if self.action_history:
				if self.action_history[-1]['type'] == 'Pen':
					for line_id in self.action_history[-1]['line_id']:
						self.canvas.delete(line_id)
				elif self.action_history[-1]['type'] == 'Line':
					if self.action_history[-1]['line_id'] is not None:
						self.canvas.delete(self.action_history[-1]['line_id'])
				elif self.action_history[-1]['type'] == 'Inpaint' or self.action_history[-1]['type'] == 'Shuffle':
					self.canvas.delete(self.action_history[-1]['image_id'])
					if len(self.reference) > 1:
						self.reference.pop()
					if len(self.inpaint_history) > 1:
						self.inpaint_history.pop()
				self.action_history.pop()
			if self.image_array_history:
				self.image_array = self.image_array_history[-1]
				self.image_array_history.pop()

	def inpaint(self):
		if self.has_input:
			######################### test using OpenCV #########################
			mask = np.any(self.image_array != self.inpaint_history[-1] , axis = 2).astype(np.uint8)
			if args.cv2:
				inpainted_image = cv2.inpaint(self.image_array , mask , 3 , cv2.INPAINT_NS)
			else:
				solver = inpainter(alpha = 0.5)
				inpainted_image = solver.inpaint(self.image_array , mask , 11)
			#####################################################################

			self.image_array_history.append(self.image_array.copy())
			self.image_array = inpainted_image.copy()
			self.inpaint_history.append(inpainted_image.copy())
			inpainted_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(inpainted_image , cv2.COLOR_BGR2RGB)))
			image_id = self.canvas.create_image(0 , 0 , anchor = NW , image = inpainted_image)
			self.reference.append(inpainted_image)
			self.action_history.append({'type' : 'Inpaint' , 'image_id' : image_id})

	def shuffle(self , x , y):
		if self.has_input:
			######################### test using OpenCV #########################
			translation_matrix = np.float32([[1 , 0 , x - self.previous_x] , [0 , 1 , y - self.previous_y]])
			shifted_image = cv2.warpAffine(self.inpaint_history[-1] , translation_matrix , (self.inpaint_history[-1].shape[1] , self.inpaint_history[-1].shape[0]))
			shifted_mask = cv2.warpAffine(self.mask.astype(np.uint8) , translation_matrix , (self.mask.shape[1] , self.mask.shape[0]))
			shifted_mask = np.repeat(np.expand_dims(shifted_mask , axis = 2) , 3 , axis = 2)
			image = (1 - shifted_mask) * self.inpaint_history[-1] + shifted_mask * shifted_image

			mask_1 = np.any(self.image_array != self.inpaint_history[-1] , axis = 2).astype(np.uint8)
			mask_2 = cv2.warpAffine(self.mask.astype(np.uint8) , translation_matrix , (self.mask.shape[1] , self.mask.shape[0]))
			mask_3 = np.logical_and(mask_1 , 1 - mask_2)
			mask_4 = cv2.erode(mask_2 , np.ones((3 , 3) , np.uint8) , iterations = 5)
			mask_5 = cv2.dilate(mask_2 , np.ones((3 , 3) , np.uint8) , iterations = 5)
			mask_6 = np.logical_and(1 - mask_4 , mask_5)
			mask = np.logical_or(mask_3 , mask_6).astype(np.uint8)

			if args.cv2:
				inpainted_image = cv2.inpaint(image , mask , 3 , cv2.INPAINT_NS)
			else:
				solver = inpainter(alpha = 0.5)
				inpainted_image = solver.inpaint(image , mask , 11)
			#####################################################################

			self.image_array = inpainted_image.copy()
			self.inpaint_history.append(inpainted_image.copy())
			inpainted_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(inpainted_image , cv2.COLOR_BGR2RGB)))
			self.canvas.itemconfig(self.action_history[-1]['image_id'] , image = inpainted_image)
			self.reference[-1] = inpainted_image

	def save_image(self):
		if self.has_input:
			file_name = askstring(title = 'Save Image' , prompt = 'Save as: ')
			if file_name is not None:
				if not (file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png')):
					showerror(title = 'Save Image' , message = 'Filename extension must be jpg, jpeg or png.')
				else:
					cv2.imwrite(file_name , self.image_array)

	def on_closing(self):
		if not self.has_input or askyesno(title = 'Quit' , message = 'Do you want to quit? Your changes will be lost if you don\'t save them.'):
			self.root.destroy()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cv2' , type = int , default = 1)
	args = parser.parse_args()
	GUI()