from PIL import Image, ImageDraw, ImageFont
import cv2
import re
import os
import numpy as np
import jieba
import itertools
import config as cfg
  
jieba.initialize() # initializng the model

class ChineseTextOverlay:
    """
    A class to overlay translated Chinese text onto an image inside a bounding box, handling text wrapping,
    font size adjustment, and positioning to ensure the text fits well within the box.
    """

    def __init__(self,img, translated_text, key, color, box) -> None:
        """
        Initialize the ChineseTextOverlay with image data, translated text, bounding box, font settings, and text color.

        Args:
            img (np.array): Image on which the text will be overlaid.
            translated_text (str): The translated text to overlay, normalized and segmented.
            key (str): Unique identifier for the text box.
            color (tuple): Color of the text.
            box (tuple): Coordinates of the bounding box where text will be placed.
        """
        self.img = img
        self.translated_text = jieba.lcut( self.normalize_punctuation(translated_text))
        self.key = key
        self.text_color = color
        self.box = box
        self.font_size= 35
        self.lowest_font_size = 18
        self.stroke_width = 3 
        self.max_font_size= 40
        self.font_path = os.path.join(cfg.FONT_PATH, "zh/font.ttf")


    def merge_exclamations(self,char_list):
        """
        Merge consecutive exclamation or special characters in the text to avoid awkward breaks.

        Args:
            char_list (list): List of characters to process.

        Returns:
            list: A list where consecutive punctuation or numeric characters are grouped.
        """
        punctuation = {'!', '?',',', ';', ':', '-', '(', ')', '[', ']', '{', '}', "'", '"'}

    # Helper function to identify punctuation or numeric characters
        def is_special_char(char):
            return char in punctuation or char.isdigit()

        grouped_list = []

        # Use itertools.groupby to group consecutive items
        for key, group in itertools.groupby(char_list, is_special_char):
            group_list = list(group)
            if key:  # if key is True, it's a group of punctuation or numeric characters
                grouped_list.append(''.join(group_list))
            else:  # otherwise, it's a group of non-special characters
                grouped_list.extend(group_list)

        return grouped_list

    def normalize_punctuation(self,text):
        """
        Convert full-width punctuation in the text to half-width for consistency.

        Args:
            text (str): Text to normalize.

        Returns:
            str: Text with normalized punctuation.
        """
        full_to_half = {
            '！': '!',
            '？': '?',
            '。': '.',
            '，': ',',
            '；': ';',
            '：': ':',
            '（': '(',
            '）': ')',
            '［': '[',
            '］': ']',
            '｛': '{',
            '｝': '}',
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'",
            "一":"|",
            "—":"|"
        }
        
        # Normalize the text
        normalized_text = ''.join(full_to_half.get(char, char) for char in text)
        return normalized_text

    # def normalize_fullwidth_to_halfwidth(self,text):
    #     normalized_text = []
    #     for char in text:
    #         code_point = ord(char)
    #         # Full-width characters are in the range U+FF01 to U+FF5E
    #         if 0xFF01 <= code_point <= 0xFF5E:
    #             # Convert full-width character to half-width character
    #             normalized_char = chr(code_point - 0xFEE0)
    #         else:
    #             normalized_char = char
    #         normalized_text.append(normalized_char)
    #     return ''.join(normalized_text)
    
    def wrap_text(self,word_list,max_text_height,canvas, log):
        """
        Wrap text to fit within the specified maximum height inside the bounding box.

        Args:
            word_list (list): List of words to wrap.
            max_text_height (int): Maximum allowable height for the wrapped text.
            canvas (ImageDraw): Drawing canvas for the text.
            log (bool): Flag to log the process.

        Returns:
            list: List of wrapped text columns.
        """
        current_line = ""
        columns = []
        font = ImageFont.truetype(self.font_path, self.font_size)



        for words in word_list:
            if "!" not in words[0] and not bool(re.search(r'\d', words)):
                sub_word_list = list(words)
                try:
                    if sub_word_list[1] == "!":
                        merge_data = "".join(sub_word_list[1:])
                        sub_word_list = [sub_word_list[0], merge_data]
                except Exception as e:
                    pass
            else:
                sub_word_list = [words]

            # Iterate over each character
            for word in sub_word_list:
                word = word.strip()
                if word in [",", "!", "!!"]:
                    font = ImageFont.truetype(self.font_path, 12)
                else:
                    font = ImageFont.truetype(self.font_path, self.font_size)

                new_line = f"{current_line}{word}\n"  # Note the change in appending the new word
                size = canvas.textbbox((0, 0), new_line.strip(), font=font)

                # Check if the new line's height is within limits
                if size[3] <= max_text_height:
                    current_line = new_line
                else:
                    # Append current line to columns
                    if current_line:
                        columns.append(current_line.strip())
                    current_line = f"{word}\n"  # Start a new line with the current word

                    if log:
                        print("New Column Added:", columns[-1])
                        print("Starting New Line with Word:", current_line)

        # Append any remaining text in current_line to columns
        if current_line:
            columns.append(current_line.strip())

        if log:
            print("Final columns:", columns)

        return columns
    
    def get_text_max_wh(self,columns,canvas,font, stroke ):
        """
            the static text 想 is put because the text which have bounding box need to be write on the single line 
            ['2017', '年', '8', '月', '7', '日', '7', '时'] in this case the text_with_max_height will be 2017 which
            has bigger text width which cause bigger spacing betteween text so the static text is set. such thar text
            width on character is used.
            the static value "想" can be any characterbecause the text width of any chinese chracter is same for the
            same font size.   
        """
        total_columns = len(columns)
        len_data = [len(text.strip().split("\n")) for text in columns]
        text_with_max_height = columns[np.argmax(len_data,axis=0)]
        if len(text_with_max_height.split("\n")) == 1:
            text_with_max_height ="想"  
        else:
            text_with_max_height = columns[np.argmax(len_data,axis=0)]

        if stroke:
            _,_, text_width, text_height = canvas.textbbox((0,0),"".join(text_with_max_height), font= font, stroke_width=self.stroke_width)
            max_text_width = (text_width*total_columns) + (self.stroke_width*total_columns)
        else:
            _,_, text_width, text_height = canvas.textbbox((0,0),"".join(text_with_max_height), font= font)
            max_text_width = (text_width*total_columns) 
        return max_text_width, text_height, text_width

    def put_text_inside(self):
        """
        Overlay the translated text inside the bounding box, adjusting font size and positioning to fit the text well.
        """
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = Image.fromarray(self.img)
        crop = self.img.crop((self.box[0],self.box[1], self.box[2], self.box[3]))
        bx,by,bw,bh = self.box[0], self.box[1],self.box[2], self.box[3]
        (x, y, w, h) = 0 ,0, crop.size[0], crop.size[1]

        padding = int(w*0.10) #int(w*0.09)
        height_padding = int (h*0.05)
        max_text_width = w - 2 * padding
        # if h>w and w >= h*0.68:
        #     thresh = 0.5
        # else:
        #     thresh = 0.85

        max_text_height = h*0.70
        canvas = ImageDraw.Draw(self.img)
        # self.translated_text = self.normalize_fullwidth_to_halfwidth(self.translated_text)
        font = ImageFont.truetype(self.font_path, self.font_size)
        self.translated_text = self.merge_exclamations(self.translated_text)
        log= False
        columns = self.wrap_text(self.translated_text, max_text_height,canvas,log)
        stroke= self.stroke_width  # if "f" in self.key else False
        text_width, text_height, character_width =self.get_text_max_wh(columns=columns, canvas=canvas, font=font, stroke=stroke )
        is_max_width_exceeed = True
        if text_width < max_text_width:
            self.font_size = 35
            self.lowest_font_size = 30
        while ( text_width > max_text_width) and self.font_size > self.lowest_font_size:
          
            self.font_size -= 1
            font = ImageFont.truetype(self.font_path,self.font_size)
            
                
            columns = self.wrap_text(self.translated_text, max_text_height, canvas, log)
            text_width, text_height,character_width =self.get_text_max_wh(columns=columns, canvas=canvas, font=font,stroke=stroke)
        while (text_height< max_text_height and text_width < max_text_width) and self.font_size <= self.max_font_size:
            self.font_size +=1
            font = ImageFont.truetype(self.font_path,self.font_size)
            columns = self.wrap_text(self.translated_text, max_text_height, canvas,log)
            text_width, text_height,character_width =self.get_text_max_wh(columns=columns, canvas=canvas, font=font,stroke=stroke)
        # print(self.font_size, self.translated_text)

        columns.reverse()
        start_x= bx + (w+ padding-text_width) //2 
        start_y =  by + abs( ((h-text_height)//2)) # ((h - text_height) // 2 )
        # if self.key == "5_0" or self.key == "4_0" or self.key=="3_0" or self.key=="5_1":
        #     print(self.key, "::" ,columns)
        for col in columns:
            col = col.replace("\n\n","\n")
            if self.text_color != (255,255,255):
            # if stroke:
                canvas.text((start_x,start_y), col, font=font, fill=self.text_color, stroke_fill=(255,255,255), stroke_width=self.stroke_width)
                # start_x = start_x + character_width 
            else:
                canvas.text((start_x,start_y), col, font=font, fill=self.text_color, stroke_fill=(0,0,0), stroke_width=self.stroke_width)
            start_x = start_x + character_width +1 # 1 is spacing between text
        
        # puting label on bounding box
        # font=ImageFont.truetype(self.font_path,25,)
        # canvas.text( (bx-10, by-20),self.key, font=font, fill=(0,0,255),stroke_fill=(255,0,0),stroke_width=2)
        
        img=  np.asarray(self.img)
        img = img[:, :, ::-1].copy()
        return img




class EnglishTextOverlay:
    """
    A class to overlay translated English text onto an image within a specified bounding box, 
    handling text wrapping, font size adjustment, and positioning to ensure optimal fit within the box.
    """

    def __init__(self,img, translated_text, key,bubble_shape, color, box) -> None:
        """
        Initialize the EnglishTextOverlay with image data, translated text, bounding box, font settings, and text color.

        Args:
            img (np.array): Image on which the text will be overlaid.
            translated_text (str): The translated text to overlay.
            key (str): Unique identifier for the text box.
            bubble_shape (str): Shape of the text bubble (e.g., Oval, Irregular).
            color (tuple): Color of the text.
            box (tuple): Coordinates of the bounding box where text will be placed.
        """
        self.img = img
        self.translated_text = translated_text
        self.key = key
        self.bubble_shape = bubble_shape
        self.text_color = color
        self.box = box
        self.font_size=25 #0.0085 * self.img.shape[0] if 0.0074 * self.img.shape[0] > 20 else 20
        self.lowest_font_size = 10
        self.stroke_width = 3 
        self.font_path = os.path.join(cfg.FONT_PATH, "en/font.ttf") #"/home/mrcloud/Downloads/mangamasterprobb/MangaMasterProBB-bold.otf" #

    def wrap_text(self,canvas,font_size, max_text_width ):
        """
        Wrap text to fit within the specified maximum width inside the bounding box.

        Args:
            canvas (ImageDraw): Drawing canvas for the text.
            font_size (int): Font size to use for wrapping.
            max_text_width (int): Maximum allowable width for the wrapped text.

        Returns:
            list: List of wrapped text lines.
        """
        words = self.translated_text.split() if isinstance(self.translated_text,str) else self.translated_text
        lines =[]
        current_line =""
        font = ImageFont.truetype(self.font_path, font_size)
        for word in words:
            new_line = f"{current_line} {word}".strip()
            size = canvas.textbbox((0,0),new_line, font=font)
            if size[2] <= max_text_width:
                current_line = new_line
            #     sub_text = list(new_line)
            #     text= ""
            #     sub_text_size = canvas.textbbox((0,0),text, font=font)
            #     if sub_text_size[2] > max_text_width:
            #         for stext in sub_text:
            #             text += stext
            #             sub_text_size = canvas.textbbox((0,0),text, font=font)
            #             if sub_text_size [2] > max_text_width:
            #                 break
            #         if len( word[len(text):]) >1:
            #             lines.append(text+"-")
            #         else: 
            #             lines.append(text)
            #         current_line = new_line[len(text):]  
            #     else:
            #         current_line = new_line
            # elif canvas.textbbox((0,0),word, font=font)[2] >max_text_width:
            #     lines.append(" ".join( new_line.split()[:-1]))
            #     sub_text =list(word)
            #     text=""
            #     sub_text_size = canvas.textbbox((0,0),text, font=font)
            #     # while sub_text_size[2] <= max_text_width:
            #     for stext in sub_text:
            #         text += stext
            #         sub_text_size = canvas.textbbox((0,0),text, font=font)
            #         if sub_text_size [2] > max_text_width:
            #             break
            #         # print(text, sub_text_size[2], max_text_width)
            #     if len( word[len(text):]) >1:
            #         lines.append(text+"-")
            #     else: 
            #         lines.append(text)
            #     # print("text:   ",text, new_line)
            #     current_line = word[len(text):]      
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line.strip())
        return lines
    
    def is_significantly_vertical_rectangle(self,width, height, threshold=0.3):
        """
        Determine if the bounding box is significantly vertical based on its width and height.

        Args:
            width (int): Width of the bounding box.
            height (int): Height of the bounding box.
            threshold (float): Ratio threshold to determine significant vertical orientation.

        Returns:
            bool: True if the rectangle is significantly vertically oriented, False otherwise.
        """
    # Calculate the minimum height to be considered significantly vertical
        min_height_for_vertical = width * (1 + threshold)
        
        # Check if the rectangle is significantly vertically oriented
        if height > min_height_for_vertical:
            return True
        else:
            return False
        
    def put_text_inside(self, padding_status):
        """
        Overlay the translated text inside the bounding box, adjusting font size, alignment, and positioning.

        Args:
            padding_status (bool): Flag to determine whether padding should be applied.

        Returns:
            np.array: Image with the overlaid text.
        """
        # print("Image shape", self.img.shape)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = Image.fromarray(self.img)
        crop = self.img.crop((self.box[0],self.box[1], self.box[2], self.box[3]))
        bx,by,bw,bh = self.box[0], self.box[1],self.box[2], self.box[3]
        (x, y, w, h) = 0 ,0, crop.size[0], crop.size[1]
        vertical=False
        # if padding_status:
        #     padding = int(w*0.265)
        # else:
        #     padding =  0 # int(w*0.13) #int(w*0.09)
        padding= int(0.2 * w)
        max_text_width = w - padding
        max_text_height = h *0.9
        canvas = ImageDraw.Draw(self.img)
        font = ImageFont.truetype(self.font_path, self.font_size)
        # print(len(self.translated_text.split()), self.translated_text, w,h,self.bubble_shape )
        smaller_dimension =min(h,2)
        if len(self.translated_text.split()) <=1 and self.is_significantly_vertical_rectangle(w,h,threshold=0.7) and self.bubble_shape != "Oval":
            vertical = True
            lines = list(self.translated_text)
            
            # lower_font_size= int(h*0.1219) if int(h*0.1219) > 13 else 13
            lower_font_size =  smaller_dimension*0.18 +1 if smaller_dimension*0.004 >= 13 else 13
            max_font_size = int(h*0.19) if int(h*0.25) <= 20 else 20
            _,_,text_width,text_height = canvas.textbbox((0,0),"\n".join(lines), font=font)
            while (text_height > (h*0.8) or text_width > max_text_width) and self.font_size >lower_font_size :
                self.font_size -= 1
                font = ImageFont.truetype(self.font_path, self.font_size)
                _,_,text_width,text_height = canvas.textbbox((0,0),"\n".join(lines), font=font)
        else:
            lines =self.wrap_text(canvas=canvas,
                                    font_size= self.font_size, 
                                    max_text_width= max_text_width
                                    )
          
            _,_,text_width,text_height = canvas.textbbox((0,0),"\n".join(lines), font=font)
            translated_word_count = len(self.translated_text.split())
            
            """ 
            if  self.is_significantly_vertical_rectangle(width=w, height=h) and translated_word_count >1:
                if  translated_word_count <=3:
                    max_text_width -= w*0.21
                elif  translated_word_count <=5:
                    if w < h*0.44:
                        max_text_width -= w*0.23
                    else:
                        max_text_width -= w*0.20
                        
                else:
                    max_text_width = w- w*0.31

            elif 0< w- text_width <= w*0.14:
                max_text_width -=  w*0.14

            """
            if (h - text_height >= 0.54* h):
                lower_font_size = text_height*0.22 if text_height*0.22 >=15 else 15
                # print(self.translated_text, lower_font_size)
            else:
                lower_font_size =  smaller_dimension*0.18 +1 if smaller_dimension*0.004 >= 11 else 11


            while (text_height > max_text_height or text_width > max_text_width) and self.font_size >lower_font_size :
                self.font_size -= 1
                # print("Font size decreased === ",self.translated_text, self.font_size,lower_font_size,text_height >  max_text_height *0.9 ,  text_width > max_text_width,self.font_size >lower_font_size,w,h,text_height)
                font = ImageFont.truetype(self.font_path, self.font_size)
                lines = self.wrap_text(canvas=canvas,
                                    font_size= self.font_size, 
                                    max_text_width= max_text_width
                                    )
                _,_,text_width,text_height = canvas.textbbox((0,0),"\n".join(lines), font=font)
            max_font_size = min(int(max_text_height * 0.28) if len(lines) <= 2 else max_text_height * 0.07, 35)
            # if max_font_size
            while text_width < max_text_width and text_height < max_text_height :
                if  self.font_size > max_font_size:
                    break
                # print("Fontsize increase  ",self.translated_text, self.font_size)
                font = ImageFont.truetype(self.font_path, self.font_size)
                lines = self.wrap_text(canvas=canvas,
                                    font_size= self.font_size, 
                                    max_text_width= max_text_width
                                    )
                _,_,text_width,text_height = canvas.textbbox((0,0),"\n".join(lines), font=font)
                self.font_size += 1

        lines = [text for text in lines if text]
        start_y = by + ((h - text_height) // 2 )
        """
        if vertical:
            start_y = by + abs(((h - text_height) // 2 ))
            # print("Vertcial true ",self.translated_text, start_y, by, h, text_height)
        elif  self.is_significantly_vertical_rectangle(w,h,threshold=0.60) and len(lines) <= 3 and self.bubble_shape != "Oval":
            if len(lines) <= 2:
                start_y = by + (text_height +h//4)
            else:
                start_y = by+(text_height//2)
            # print(self.translated_text, self.bubble_shape, self.font_size, start_y, by)
        # elif not self.is_significantly_vertical_rectangle(w,h,threshold=0.60) and self.bubble_shape == "Irregular":
        #     start_y = by + ((h - text_height)//2)  + h*0.15
        #     print(lines, start_y)
        else:
            start_y = by + ((h - text_height) // 2 )
        """
        # while (start_y+text_height) >(bh - bh*0.01) and start_y > by :
        #     start_y -= 1
            
        
        # print(self.translated_text, self.bubble_shape, self.font_size,vertical, lower_font_size,max_font_size)
        for line in lines:
            _,_,text_width,text_height = canvas.textbbox((0,0),line, font=font)
            # if self.is_significantly_vertical_rectangle(w,h):
            #     start_x =(w-text_width)//2   if (bx  + (w-text_width) //2) > 0 else 0
            # else:
            start_x =  bx  + ((w-text_width) //2)  if (bx  + (w-text_width) //2) > 0 else 0
            
            if "f" in self.key:
                canvas.text((start_x,start_y), line, font=font, fill=self.text_color, stroke_fill=(255,255,255), stroke_width=4)
                start_y += text_height + 6
            else:
                # draw_outline(line,start_x,start_y,canvas, font)
                if self.text_color != (255,255,255):
                    canvas.text((start_x,start_y), line, font=font, fill=self.text_color,stroke_fill=(255,255,255), stroke_width=3)
                else: 
                    canvas.text((start_x,start_y), line, font=font, fill=self.text_color,stroke_fill=(0,0,0), stroke_width=3)

                if vertical:
                    start_y += text_height +1
                else:
                    start_y += text_height + 5
        img=  np.asarray(self.img)
        img = img[:, :, ::-1].copy()

        return img
        
