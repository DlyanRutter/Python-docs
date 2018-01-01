import turtle

window = turtle.Screen()
window.bgcolor("blue")
irwin = turtle.Turtle()
irwin.shape("turtle")
irwin.speed(3)


def biggest():
    irwin.setpos(-50,50)
    irwin.clear()
    i = 0
    while i < 3:
        irwin.forward(100)
        irwin.right(120)
        i = i + 1

def mid_size():          
    irwin.up()
    irwin.setpos(0,50)
    irwin.pd()    
    irwin.left(120)
    irwin.forward(60)
    irwin.right(120)
    irwin.forward(60)
    irwin.right(120)
    irwin.forward(60)

    irwin.up()
    irwin.setpos(22,5)
    irwin.pd()    
    irwin.left(120)
    irwin.forward(60)
    irwin.right(120)
    irwin.forward(60)
    irwin.right(120)
    irwin.forward(60)

    irwin.up()
    irwin.setpos(-22,5)
    irwin.pd()    
    irwin.left(120)
    irwin.forward(60)
    irwin.right(120)
    irwin.forward(60)
    irwin.right(120)
    irwin.forward(60)

    

print biggest()
print mid_size()
