cdef extern from "opencv2/opencv.hpp" namespace "cv":
    cdef cppclass Rect_[T]:
        Rect_() except +
        Rect_(T, T, T, T) except +
        T area()
    cdef Rect_[T] operator&[T](Rect_[T], Rect_[T])

def is_rects_overlap(left_rect, right_rect):
    """
    left_rects: list of int: [x, y, width, height]
    right_rects: list of int: [x, y, width, height]
    """
    # I find that Rect_[int](*left_rect) can not work, so I have define these
    # variables below manually.
    cdef int left_rect_x = left_rect[0]
    cdef int left_rect_y = left_rect[1]
    cdef int left_rect_width = left_rect[2]
    cdef int left_rect_height = left_rect[3]
    cdef Rect_[int] c_left_rect = Rect_[int](
            left_rect_x, left_rect_y,
            left_rect_width, left_rect_height)
    cdef int right_rect_x = right_rect[0]
    cdef int right_rect_y = right_rect[1]
    cdef int right_rect_width = right_rect[2]
    cdef int right_rect_height = right_rect[3]
    cdef Rect_[int] c_right_rect = Rect_[int](
            right_rect_x, right_rect_y,
            right_rect_width, right_rect_height)
    cdef Rect_[int] overlap_rect = c_left_rect & c_right_rect
    # OpenCV doc says: Negative samples are taken from arbitrary images.
    # These images must not contain detected objects.
    # So no overlap!
    # However, qinjian stand for restricted overlap does not mean contain,
    # So change to critical_rate again
    cdef double overlap_rate = 1.0 * overlap_rect.area() / c_right_rect.area()
    # Note that area() return int
    cdef double critical_rate = 0.5
    return overlap_rate > critical_rate
