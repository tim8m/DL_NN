import numpy as np

def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    Численная проверка градиента в num_checks случайных точках.
    f: функция, принимающая x и возвращающая скаляр (loss)
    x: матрица параметров (здесь W)
    analytic_grad: аналитический градиент той же формы, что x
    """
    flat_x = x.ravel()
    flat_grad = analytic_grad.ravel()

    # случайные индексы
    ix = np.random.choice(flat_x.size, num_checks, replace=False)

    for i in ix:
        old_val = flat_x[i]

        # f(x + h)
        flat_x[i] = old_val + h
        fxph = f(x)

        # f(x - h)
        flat_x[i] = old_val - h
        fxmh = f(x)

        # восстановить
        flat_x[i] = old_val

        # численная производная (центральная разность)
        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = flat_grad[i]

        rel_error = abs(grad_numerical - grad_analytic) / \
                    max(1e-8, abs(grad_numerical) + abs(grad_analytic))

        print(f'[{i:4d}] numerical: {grad_numerical: .8e}  '
              f'analytic: {grad_analytic: .8e}  '
              f'relative error: {rel_error:.3e}')
