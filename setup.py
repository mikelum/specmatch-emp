import ez_setup
ez_setup.use_setuptools()

import os, sys
from setuptools import setup, find_packages

on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    setup(
        name="SpecMatch-Emp",
        version="0.2",
        packages=find_packages(),
        include_package_data=True
    )
else:
    setup(
        name="SpecMatch-Emp",
        version="0.2",
        packages=find_packages(),
        install_requires=[
            "numpy",
            "scipy",
            "matplotlib",
            "h5py",
            "pandas",
            "lmfit",
            "astropy",
            "astroquery",
            "isochrones",
        ],
        include_package_data=True,
        entry_points={
            'console_scripts': [
                'smemp=specmatchemp.cli:main'
            ],
        },
    )

    # download library
    HOMEDIR = os.environ['HOME']
    SPECMATCHDIR = "{0}/.specmatchemp/".format(HOMEDIR)
    if not os.path.exists(SPECMATCHDIR):
        os.mkdir(SPECMATCHDIR)

    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = "\r%5.1f%% %*d / %d" % (
                percent, len(str(totalsize)), readsofar, totalsize)
            sys.stderr.write(s)
            if readsofar >= totalsize:  # near the end
                sys.stderr.write("\n")
        else:  # total size is unknown
            sys.stderr.write("read %d\n" % (readsofar,))

    # liburl = "https://zenodo.org/record/60225/files/library.h5"
    liburl = "https://www.dropbox.com/s/po0kzgjn1j9ha2v/library.h5#"
    LIBPATH = os.path.join(SPECMATCHDIR+"library.h5")
    if not os.path.exists(LIBPATH):
        os.system("wget --no-check-certificate --output-document=${HOME}/.specmatchemp/library.h5 https://www.dropbox.com/s/po0kzgjn1j9ha2v/library.h5#")

    # Additional csv files
    csv_urls = ["https://www.dropbox.com/s/sdeouxrs7jrvss9/hires_telluric_mask.csv#",
                "https://www.dropbox.com/s/wmdn6z67op2est0/detrend.csv#",
                "https://www.dropbox.com/s/8ookg989hkyx7k1/uncertainties.csv#"]
    for f in csv_urls:
        outfile = os.path.join(SPECMATCHDIR,f.split('/')[-1][:-1])
        cmd = "wget --no-check-certificate {} -O {}".format(f,outfile)
        os.system(cmd)

    specdir = os.path.join(SPECMATCHDIR, 'spectra')
    if not os.path.exists(specdir):
        os.mkdir(specdir)
        # download references
        ref_urls = ["https://www.dropbox.com/s/i397kkebdm2b5ez/nso.fits#",
                    "https://www.dropbox.com/s/6oim1suxnu3mci8/rj26.532.fits#",
                    "https://www.dropbox.com/s/cepbxprp6wvdmee/rj72.718.fits#",
                    "https://www.dropbox.com/s/9pe7pwae489mjdw/rj59.1926.fits#"]
        for ref in ref_urls:
            outfile = os.path.join(specdir,ref.split('/')[-1][:-1])
            cmd = "wget --no-check-certificate {} -O {}".format(ref,outfile)
            os.system(cmd)


#        from six.moves import urllib
#        print("Downloading library.h5")
#        urllib.request.urlretrieve(liburl, LIBPATH, reporthook)




