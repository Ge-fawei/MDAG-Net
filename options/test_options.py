from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False

        self.parser.add_argument('--results_dir', type=str, default='./results/', help='Saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='Aspect ratio of result images')

        self.parser.add_argument('--which_epoch', required=True, type=int, help='Which epoch to load for inference')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc '
                                                                           '(determines name of folder to load from)')

        self.parser.add_argument('--how_many', type=int, default=50, help='How many test images to run '
                                                                          '(if serial_test not enabled)')
        self.parser.add_argument('--serial_test', action='store_true', help='Read each image once from folders '
                                                                            'in sequential order')

        self.parser.add_argument('--which_slice', type=int, default=999,
                                 help='Which slice of images to be test, not specified if testing on RobotCar dataset')
        self.parser.add_argument('--test_using_cos', action='store_true',
                                 help='Test the model with cosine or mean cosine metric, use L2 otherwise')

        self.parser.add_argument('--use_two_stage', action='store_true',
                                 help='Use the from coarse to fine two-stage strategy')
        self.parser.add_argument('--top_n', type=int, default=1,
                                 help='Top n candidates for the finer retrieval in the two-stage strategy')


