with import <nixpkgs> {};
let

    observations = python3Packages.buildPythonPackage rec {
        pname = "Observations";
        version = "0.1.4";

        src = builtins.fetchTarball {
			url = "https://files.pythonhosted.org/packages/f3/e8/f219dc1b23b0ebdface7a3d80fa907771cd911ea8f641bf0b8b03201830e/observations-0.1.4.tar.gz";
            sha256 = "113w9d4ddr696507x2dxdr2qkqxqd189p6zz3ra6flgrh9w00li8";
        };

        propagatedBuildInputs = with python3Packages; [ six numpy requests ];

        doCheck = false;

    };


	environment = (python3.withPackages (ps: [ ps.numpy observations ]));

in mkShell {
    buildInputs = [ environment ];
}
