import numpy as np



#----------------------------------------decorate sample---------------------------------------------------
def decorate(Yosef):
    print('Or')
    Yosef()
    print('Cohen')

@decorate
def NotMatterWhatIsTheName():
    print('Yosef')

#--------------------------------------------section 2-----------------------------------------------------
# class Point:
#     def __init__(self):
#         self._X = 0
#         self._Y = 0
#
#     def __str__(self):
#         return '(' + str(self._X) + ',' + str(self._Y) + ')'
#
#     @property
#     def get_X(self):
#         return self._X
#
#     @property
#     def get_Y(self):
#         return self._Y
#
#     @get_X.setter
#     def set_X(self, a):
#         self._X = a
#
#     @get_Y.setter
#     def set_Y(self, a):
#         self._Y = a
#
#     def distance(self, other):
#         d = Point()
#         d.set_X = other.get_X - self._X
#         d.set_Y = other.get_Y - self._Y
#         return d
#
#
# class Line(Point):
#     def __init__(self, x, y):
#         super()._init_()
#         self.x = x
#         self.y = y
#
#     def distance(self):
#         return self.x.distance(self.y)
#
#     def on_line(self, other):
#         delta_x = self.distance().get_X
#         delta_y = self.distance().get_Y
#         m = delta_y / delta_x
#         n = self.x.get_Y - self.x.get_X * m
#         if other.get_X * m + n == other.get_Y:
#             return print('The point {} is on the line'.format(other))
#         else:
#             return print('The point {} is not on the line'.format(other))
#
#
# t = Point()
# # t.set_Y = 1
# # t.set_X = 1
# # g = Point()
# # g.set_Y = 4
# # g.set_X = 3
# # print(t.distance(g))
# # l = Line(t, g)
# # f = Point()
# # f.set_X = 5
# # f.set_Y = 5
# # print(l.on_line(t))
# # print(l.on_line(f))
#
#
# # Section 3
# class shape:
#     def __init__(self, point_list):
#         self.point_list = point_list
#         self.rect_edge = np.unique(self.point_list)
#         self.calc_area = 'you did not calculate the area yet'
#         self.calc_perimeter = 'you did not calculate the perimeter yet'
#
#     def __str__(self):
#         return 'area:' + str(self.calc_area) + '\n' 'perimeter:' + str(self.calc_perimeter)
#
#     def area(self):
#         self.calc_area = self.rect_edge[0] * self.rect_edge[1]
#         return self.calc_area
#
#     def perimeter(self):
#         self.calc_perimeter = 2 * (self.rect_edge[0] + self.rect_edge[1])
#         return self.calc_perimeter
#
#
# class Triangle(shape):
#     def __init__(self, point_list):
#         super().__init__(point_list)
#         self.point_list = point_list
#
#     def __str__(self):
#         return 'area:' + str(self.calc_area) + '\n' 'perimeter:' + str(self.calc_perimeter)
#
#     def area(self):
#         sorted_edges = np.sort(np.array(self.point_list))
#         pitagoras = sorted_edges[0]*2 + sorted_edges[1]*2
#         if pitagoras == sorted_edges[2]**2:
#             self.calc_area = (sorted_edges[0] * sorted_edges[1])/2
#             return self.calc_area
#         else:
#             return print('The triangle is not Right-angled triangle ')
#
#     def perimeter(self):
#         self.calc_perimeter = np.sum(np.array(self.point_list))
#         return self.calc_perimeter
#
#
# class Rectangle (shape):
#     def __init__(self, point_list):
#         super().__init__(point_list)
#         # self.list_of_points = list_of_points
#
#     def __str__(self):
#         return 'Area:' + str(self.calc_area) + '\n' 'Perimeter:' + str(self.calc_perimeter) + '\n' 'List of ' \
#               'Points:' + \
#                str(self.point_list)
#
#
#
# # d = shape([1, 4, 1, 4])
# # d.area()
# # d.perimeter()
# # print(d)
# #
# # f = Rectangle([4, 4, 3, 3])
# # f.area()
# # f.perimeter()
# # print(f)
#
# # j = Triangle([3, 4, 5])
# # j.area()
# # j.perimeter()
# # print(j)