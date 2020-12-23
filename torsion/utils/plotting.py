import io
import matplotlib.pyplot as plt
from openeye import oechem, oedepict
from torsion.utils.process_sd_data import get_sd_data
from torsion.dihedral import get_dihedral
from IPython.display import Image, HTML

font = {'family': 'Arial',
        'weight': 'normal',
        'size': 14}
plt.rc('font', **font)

def plot_indices(mol2, width=200, height=200):
    mol = mol2.CreateCopy()

    opts = oedepict.OE2DMolDisplayOptions(width, height, oedepict.OEScale_AutoScale)
    opts.SetAtomPropertyFunctor(oedepict.OEDisplayAtomIdx())
    oedepict.OEPrepareDepiction(mol)

    disp = oedepict.OE2DMolDisplay(mol, opts)
    img = oedepict.OEImage(width, height)

    oedepict.OERenderMolecule(img, disp)
    return img


def plot_dihedral(mol2, width=200, height=200):
    mol = mol2.CreateCopy()
    dihedralAtomIndices = [
        int(x) - 1 for x in get_sd_data(mol, "TORSION_ATOMS_FRAGMENT").split()
    ]
    dih, tor = get_dihedral(mol, dihedralAtomIndices)

    opts = oedepict.OE2DMolDisplayOptions(width, height, oedepict.OEScale_AutoScale)
    oedepict.OEPrepareDepiction(mol)

    disp = oedepict.OE2DMolDisplay(mol, opts)
    img = oedepict.OEImage(width, height)

    hstyle = oedepict.OEHighlightByBallAndStick(oechem.OEBlueTint)
    oedepict.OEAddHighlighting(disp, hstyle, dih)
    hstyle = oedepict.OEHighlightByColor(oechem.OERed)
    oedepict.OEAddHighlighting(disp, hstyle, tor)

    oedepict.OERenderMolecule(img, disp)
    return img


def oenb_draw_dihedral(mol2, width=200, height=200):
    img = plot_dihedral(mol2, width=width, height=height)
    return Image(data=oedepict.OEWriteImageToString("png", img))


def highlight_atoms_in_mol(mol2, dihedralAtomIndices, width=200, height=200):
    mol = mol2.CreateCopy()
    opts = oedepict.OE2DMolDisplayOptions(width, height, oedepict.OEScale_AutoScale)
    oedepict.OEPrepareDepiction(mol)

    disp = oedepict.OE2DMolDisplay(mol, opts)
    img = oedepict.OEImage(width, height)

    hstyle = oedepict.OEHighlightByBallAndStick(oechem.OEBlueTint)
    for atom_idx in dihedralAtomIndices:
        oedepict.OEAddHighlighting(disp, hstyle, oechem.OEHasAtomIdx(atom_idx))

    oedepict.OERenderMolecule(img, disp)
    return img


def draw_subsearch_highlights(mol, subsearch, width=400.0, height=400.0):
    """
    Draws the hits for the substructure in a given molecule.
    
    Copied from http://notebooks.eyesopen.com/substructure-search-pandas-oenotebook.html
    """
    opts = oedepict.OE2DMolDisplayOptions(width, height, oedepict.OEScale_AutoScale)

    mol = oechem.OEGraphMol(mol)
    oedepict.OEPrepareDepiction(mol)
    img = oedepict.OEImage(width, height)
    hstyle = oedepict.OEHighlightByBallAndStick(oechem.OEBlueTint)

    disp = oedepict.OE2DMolDisplay(mol, opts)
    unique = True
    for match in subsearch.Match(mol, unique):
        oedepict.OEAddHighlighting(disp, hstyle, match)

    oedepict.OERenderMolecule(img, disp)
    # return oenb.draw_oeimage_to_img_tag(img)
    return img


# The following code is from https://mindtrove.info/jupyter-tidbit-image-gallery/
def _src_from_data(data):
    """Base64 encodes image bytes for inclusion in an HTML img element"""
    img_obj = Image(data=data)
    for bundle in img_obj._repr_mimebundle_():
        for mimetype, b64value in bundle.items():
            if mimetype.startswith('image/'):
                return f'data:{mimetype};base64,{b64value}'

def nb_gallery(images, row_height='auto'):
    """Shows a set of images in a gallery that flexes with the width of the notebook.
    
    Parameters
    ----------
    images: list of str or bytes
        URLs or bytes of images to display

    row_height: str
        CSS height value to assign to all images. Set to 'auto' by default to show images
        with their native dimensions. Set to a value like '250px' to make all rows
        in the gallery equal height.
    """
    figures = []
    for image in images:
        if isinstance(image, bytes):
            src = _src_from_data(image)
            caption = ''
        else:
            src = image
            caption = f'<figcaption style="font-size: 0.6em">{image}</figcaption>'
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{src}" style="height: {row_height}">
              {caption}
            </figure>
        ''')
    return HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')

def plot_energy_profile(theta, E):
    fig = plt.figure(figsize=(2.75,2.25))
    ax = fig.add_subplot(111)
    ax.plot(theta, E, '.')
    ax.set_xticks([-90, 0, 90])
    ax.set_xlim([-180, 180])
    ax.set_xlabel(r"$\theta (^o)$")
    ax.set_ylabel("E")
    fig.tight_layout()
    plt.close()
    return fig

def grab_plot_as_png(f):
    pic_IObytes = io.BytesIO()
    f.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    return pic_IObytes.read()