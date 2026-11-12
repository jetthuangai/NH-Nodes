"""Which folders the image nodes and routes of NH-Nodes may read and write.

A workflow is data: it can come from anyone, and ComfyUI can be reachable from the network. So the folder text of a
node (or of a request) only works inside

* reading: ComfyUI's ``input``, ``output`` and ``temp`` folders,
* writing: ComfyUI's ``output`` folder,

and the folders the owner of the machine listed on disk under ``nh_image_folders`` in ``extra_model_paths.yaml``
(both reading and writing)::

    nh_nodes_images:
        nh_image_folders: |
            D:/Photos/
            E:/Exports/

A path is checked twice. First as written (``..`` resolved, no disk access), so a path outside every allowed folder
is refused before anything touches it: a ``\\\\server\\share`` path never makes Windows contact that server. Then
with links and junctions resolved, so a link inside an allowed folder cannot lead out of it.
"""

import os

import folder_paths

FOLDER_KEY = "nh_image_folders"

_declared = []


class FolderNotAllowed(PermissionError):
    """The path lies outside every folder NH-Nodes may use. An OSError, so callers that skip unreadable paths skip it."""


def _absorb_declared():
    # ComfyUI turns every key of extra_model_paths.yaml into a model folder name, and its /models routes list the
    # files of every such name. These folders hold photos, not models: take the locations and drop the name.
    entry = folder_paths.folder_names_and_paths.pop(FOLDER_KEY, None)
    if entry:
        for path in entry[0]:
            if path and path not in _declared:
                _declared.append(path)


def declared_folders():
    """The folders listed under ``nh_image_folders``; [] when the key is not used."""
    _absorb_declared()
    return list(_declared)


def read_roots():
    return [
        folder_paths.get_input_directory(),
        folder_paths.get_output_directory(),
        folder_paths.get_temp_directory(),
        *declared_folders(),
    ]


def write_roots():
    return [folder_paths.get_output_directory(), *declared_folders()]


def _contains(root, path):
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:  # another drive, or a drive path against a network share
        return False


def _written(path):
    return os.path.normcase(os.path.abspath(path))


def _resolved(path):
    return os.path.normcase(os.path.realpath(path))


def is_inside(path, roots):
    """True when ``path`` is one of ``roots`` or below one, both as written and with links resolved."""
    roots = [root for root in roots if root]
    resolved_roots = [_resolved(root) for root in roots]  # the roots are the owner's: resolving them is safe
    written = _written(path)
    # As written, the path must lie in a root under either of the root's spellings (a root that is itself a link
    # may be typed by its real location).
    if not any(_contains(root, written) for root in [*map(_written, roots), *resolved_roots]):
        return False
    resolved = _resolved(path)
    return any(_contains(root, resolved) for root in resolved_roots)


def _refusal(path, what, action, defaults):
    return FolderNotAllowed(
        f"NH-Nodes: {what} is outside the folders NH-Nodes may {action}: {str(path)!r}. Allowed are {defaults} and "
        f"the folders listed under '{FOLDER_KEY}' in extra_model_paths.yaml. To allow another folder, add it there, "
        f"for example\n    nh_nodes_images:\n        {FOLDER_KEY}: D:/Photos/\nand restart ComfyUI."
    )


def require_readable(path, what="The folder"):
    """``path`` as an absolute path; FolderNotAllowed (naming ``what``) when NH-Nodes may not read there."""
    if not is_inside(path, read_roots()):
        raise _refusal(path, what, "read", "ComfyUI's input, output and temp folders")
    return os.path.abspath(path)


def require_writable(path, what="The folder"):
    """``path`` as an absolute path; FolderNotAllowed (naming ``what``) when NH-Nodes may not write there."""
    if not is_inside(path, write_roots()):
        raise _refusal(path, what, "write to", "ComfyUI's output folder")
    return os.path.abspath(path)


def can_read(path):
    return is_inside(path, read_roots())


_absorb_declared()
