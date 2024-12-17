from scipy._lib._util import _RichResult

__all__ = ["ImagingResult"]


class ImagingResult(_RichResult):
    """
    Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    img : ndarray
        The reconstructed final image
    success : bool
        Whether or not the method converged.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    cost_fun : float
        Value of objective function at `x`.
    cost_fun_log : ndarray
        Value of the object function at nit iterations.
    residue : float
        Discrepancy between x and b.
    nit : int
        Number of iterations performed by the optimizer.

    Notes
    -----
    Depending on the specific solver being used, `OptimizeResult` may
    not have all attributes listed here, and they may have additional
    attributes not listed here. Since this class is essentially a
    subclass of dict with attribute accessors, one can see which
    attributes are available using the `OptimizeResult.keys` method.

    """
    pass
