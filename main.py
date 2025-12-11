import sys
import requests
import markdown
import json
from PyQt6.QtWidgets import (
   QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
   QScrollArea, QFrame, QLabel, QSizePolicy, QLayout
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize

MODEL_NAME = "deepseek-r1:32b"
OLLAMA_URL = "http://localhost:11434/api/chat"


class OllamaWorker(QThread):
   result_ready = pyqtSignal(str)
   error = pyqtSignal(str)
   
   def __init__(self, prompt):
      super().__init__()
      self.prompt = prompt
      self.stop_requested = False
      self.session = requests.Session()
   
   def run(self):
      if self.stop_requested:
         return
      
      try:
         payload = {
               "model": MODEL_NAME,
               "messages": [{"role": "user", "content": self.prompt}],
               "stream": False
         }
         
         # IMPORTANT: Use stream=True so we can abort mid-response
         with self.session.post(
               OLLAMA_URL,
               json=payload,
               stream=True,
               timeout=600
         ) as r:
         
               # If stopped before headers arrive
               if self.stop_requested:
                  r.close()
                  return
               
               r.raise_for_status()
               
               # Accumulate response body with cancellation checks
               chunks = []
               for chunk in r.iter_content(chunk_size=1024):
                  if self.stop_requested:
                     r.close()
                     return
                  if chunk:
                     chunks.append(chunk)
               
               data = b"".join(chunks).decode("utf-8")
               data = json.loads(data)
               
               if not self.stop_requested:
                  self.result_ready.emit(data["message"]["content"])
      
      except Exception as e:
         if not self.stop_requested:
               self.error.emit(str(e))
   
   def stop(self):
      """Gracefully stop the worker."""
      self.stop_requested = True
      try:
         self.session.close()  # This forces any ongoing requests to abort
      except Exception:
         pass


class Bubble(QFrame):
   def __init__(self, text, sender):
      super().__init__()
      self.sender = sender
      
      self.setStyleSheet(f"""
         QFrame {{
            background-color: {'#e0e0e0' if sender == 'AI' else '#d1f0ff'};
            color: #000;
            border-radius: 12px;
            padding: 10px 14px;
         }}
      """)
      
      layout = QVBoxLayout(self)
      layout.setContentsMargins(0, 0, 0, 0)
      layout.setSpacing(2)
      
      # IMPORTANT: set the layout to recalc size based on minimum size
      self.setMinimumWidth(50)   # Minimum width
      self.setMaximumWidth(400)   # Maximum width
      layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
      
      # Text label
      self.text_label = QLabel(text)
      self.text_label.setWordWrap(True)
      self.text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
      self.text_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
      layout.addWidget(self.text_label)
      
      # Sender label
      sender_label = QLabel(sender)
      sender_label.setStyleSheet("font-size:10px; color:#555;")
      sender_label.setAlignment(Qt.AlignmentFlag.AlignLeft if sender == 'AI' else Qt.AlignmentFlag.AlignRight)
      sender_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
      layout.addWidget(sender_label)


class ChatWindow(QWidget):
   def __init__(self):
      super().__init__()
      self.setWindowTitle("Chatbot")
      self.resize(650, 800)
      
      main_layout = QVBoxLayout(self)
      main_layout.setContentsMargins(10, 10, 10, 10)
      main_layout.setSpacing(8)
      
      # Scroll area for chat
      self.scroll_area = QScrollArea()
      self.scroll_area.setWidgetResizable(True)
      self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
      main_layout.addWidget(self.scroll_area)
      
      self.chat_container = QWidget()
      self.chat_layout = QVBoxLayout(self.chat_container)
      self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
      self.scroll_area.setWidget(self.chat_container)
      
      # Input area
      input_layout = QHBoxLayout()
      self.input_line = QLineEdit()
      self.input_line.setStyleSheet("""
            QLineEdit {
               font-size:16px;
               padding:16px 20px;
               border-radius:8px;
               border:1px solid #ccc;
            }
         """)
      self.input_line.setPlaceholderText("Type your message...")
      self.send_btn = QPushButton()
      self.send_btn.setIcon(QIcon("./assets/send_icon.png"))
      self.send_btn.setIconSize(QSize(24, 24))
      self.send_btn.setStyleSheet("""
            QPushButton {
               background: white;
               color: black;
               font-size:18px;       /* larger text */
               padding:14px 24px;    /* larger button area */
               border-radius:10px;   /* optional for bigger radius */
               border:1px solid #aaa;
            }
            QPushButton:hover {
               background: #f0f0f0;
            }
         """)
      input_layout.addWidget(self.input_line)
      input_layout.addWidget(self.send_btn)
      main_layout.addLayout(input_layout)
      
      # Typing animation
      self.typing_timer = QTimer()
      self.typing_timer.timeout.connect(self.type_next_char)
      self.pending_text = ""
      self.typing_index = 0
      self.current_bubble_label = None
      
      # Connections
      self.ai_worker = None
      self.ai_thinking = False
      self.send_btn.clicked.connect(self.handle_send_or_stop)
      self.input_line.returnPressed.connect(self.handle_send_or_stop)
   
   def add_bubble(self, text, sender):
      bubble = Bubble(text, sender)
      align = Qt.AlignmentFlag.AlignLeft if sender == "AI" else Qt.AlignmentFlag.AlignRight
      self.chat_layout.addWidget(bubble, alignment=align)
      self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
      return bubble
   
   def handle_send_or_stop(self):
      if self.ai_thinking:
         # Stop AI processing
         if self.ai_worker and self.ai_worker.isRunning():
               self.ai_worker.stop()  # stops the OllamaWorker
         self.typing_timer.stop()        # stop any ongoing typing
         if self.current_bubble_label:
               # Complete the bubble text immediately
               self.current_bubble_label.text_label.setText(self.pending_text)
         self.ai_thinking = False
         self.send_btn.setIcon(QIcon("./assets/send_icon.png"))
         return
      
      # Otherwise, normal send
      msg = self.input_line.text().strip()
      if not msg:
         return
      self.add_bubble(msg, "User")
      self.input_line.clear()
      
      # Start AI processing
      self.ai_worker = OllamaWorker(msg)
      self.ai_worker.result_ready.connect(self.start_typing_animation)
      self.ai_worker.error.connect(lambda e: self.add_bubble(f"Error: {e}", "AI"))
      self.ai_worker.start()
      self.ai_thinking = True
      self.send_btn.setIcon(QIcon("./assets/stop_icon.png"))
   
   def start_typing_animation(self, text):
      if not text:
         # AI returned empty, no bubble
         self.ai_thinking = False
         self.send_btn.setIcon(QIcon("./assets/send_icon.png"))
         return
      
      # Bubble with "..." for AI
      bubble = Bubble("...", "AI")
      self.current_bubble_label = bubble
      self.chat_layout.addWidget(bubble, alignment=Qt.AlignmentFlag.AlignLeft)
      self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
      self.pending_text = text
      self.typing_index = 0
      self.current_bubble_label.text_label.setText("")
      self.typing_timer.start(30)
   
   def type_next_char(self):
      if self.typing_index < len(self.pending_text):
         label = self.current_bubble_label.text_label
         label.setText(label.text() + self.pending_text[self.typing_index])
         self.typing_index += 1
         label.adjustSize()
         self.current_bubble_label.adjustSize()
         self.chat_container.adjustSize()
         self.scroll_area.ensureWidgetVisible(self.current_bubble_label)
      else:
         self.typing_timer.stop()
         self.ai_thinking = False
         self.send_btn.setIcon(QIcon("./assets/send_icon.png"))




if __name__ == "__main__":
   app = QApplication(sys.argv)
   w = ChatWindow()
   w.show()
   sys.exit(app.exec())
