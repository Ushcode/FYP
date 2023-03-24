 #!/usr/bin/python


# Save to a file

f= open("test.txt","w+")

for i in range(10):
    f.write("This is line %d\r\n" % (i+1))

f= open("test.txt", "a+")

for i in range(2):
    f.write("Appended line %d\r\n" % (i+1))
f.close()

# Save image to pdf

import matplotlib.pyplot as plt

f = plt.figure()
plt.plot(range(10), range(10), "o")
plt.show()

f.savefig("foo.pdf", bbox_inches='tight')


from matplotlib import pyplot as plt

fig, ax = plt.subplots() # or
fig.savefig('filename.eps', format='eps')
