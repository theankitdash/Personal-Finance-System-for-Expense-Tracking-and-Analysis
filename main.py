from kivy.app import App
from kivy.uix.label import Label

class TutoringApp(App):
    def build(self):
        return Label(text='Welcome to the Tutoring System!')

if __name__ == '__main__':
    TutoringApp().run()