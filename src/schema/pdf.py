from dataclasses import dataclass

@dataclass
class PDF:
    pages: list[str]
    name: str
    type: str

    def __init__(self,name):
        self.name = name
        self.pages = []
        self.type = "unknown"

    def add_page(self,page):
        self.pages.append(page)

    def __str__(self):
        full_text = f"\n############################# begin of {self.name} of type {self.type} #############################\n"
        for page in self.pages:
            full_text += str(page)
        full_text += f"\n############################# end of {self.name} of type {self.type} #############################\n"
        return full_text