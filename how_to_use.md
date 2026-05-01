# UniSparse Toolbox Packaging and User Guide

This guide explains how to:

1. Package this project as a MATLAB toolbox (`.mltbx`) so users do not need to run `init_project`.
2. Install and use the toolbox as an end user.
3. Update and troubleshoot the toolbox.

---

## 1. Why toolbox packaging is the right approach

Without toolbox packaging, MATLAB can only find `unisparse` if:

- the user is currently in the folder containing `unisparse.m`, or
- that folder is manually added to the MATLAB path.

A toolbox install solves this once, persistently:

- users install one file (`.mltbx`),
- required folders are added automatically,
- users call `unisparse(...)` directly in any session,
- no need to run `init_project` each time.

---

## 2. Project structure assumptions

This guide assumes your current structure includes:

- `Unisparse/`
- `supp funs/`
- `other methods/`
- `Data generation/`
- `RMPSH/`
- main function: `Unisparse/unisparse.m`

That structure is fine for toolbox packaging.

---

## 3. One-time author setup (you, the seller)

### Step 1: Open project in MATLAB

1. Open MATLAB.
2. Set the current folder to the project root.

### Step 2: Create a toolbox project

Use MATLAB GUI:

1. Go to **Home > Add-Ons > Package Toolbox**.
2. Choose **Package a Folder** (or similar wording, depending on MATLAB version).
3. Select your project root folder.

MATLAB creates a toolbox project configuration (for example, a `.prj` file).

### Step 3: Fill toolbox metadata

In the toolbox packaging window, set:

- **Toolbox name**: `UniSparse`
- **Version**: e.g. `1.0.0`
- **Summary**: short one-line description
- **Description**: detailed text (methods, requirements)
- **Author/Company**: your details

### Step 4: Include required files/folders

Ensure these are included:

- `Unisparse/**`
- `supp funs/**`
- `other methods/**`
- `RMPSH/**`

Optional include:

- `Data generation/**` (if you want users to run your data generation helpers)
- demos/tests/scripts

Avoid including temporary output files (`*.mat`, logs, caches) unless required.

### Step 5: Configure toolbox paths

In toolbox settings, add folders to MATLAB path on installation:

- `Unisparse`
- `supp funs`
- `other methods`
- `RMPSH`
- optionally `Data generation`

This is the key part that removes the need for `init_project`.

### Step 6: Set startup/shutdown actions (optional)

Optional startup action:

- You can add a startup function that validates environment or displays a welcome message.
- Usually not required if your entry function handles runtime checks.

Optional shutdown action:

- Remove temporary resources if needed.

### Step 7: Add documentation and examples

Strongly recommended:

- include this file (`how_to_use.md`) or PDF documentation,
- include a minimal example script:

```matlab
% Example usage after toolbox installation
[X, y, ~, ~, ~] = Generate_data_scenario_homecourt(120, 10);
results = unisparse(X, y, [1e-3, 1e1], 5, 'all');
```

### Step 8: Build the toolbox

In the packaging tool, click **Package**.

Output will be a file like:

- `UniSparse.mltbx`

Distribute this `.mltbx` file to users.

---

## 4. End-user installation steps (buyer side)

Share these exact steps with your users.

### Install

1. Download `UniSparse.mltbx`.
2. In MATLAB, double-click the file, or use:

```matlab
matlab.addons.install('C:/path/to/UniSparse.mltbx')
```

3. MATLAB installs the toolbox and updates path automatically.

### Verify installation

Run:

```matlab
which unisparse -all
```

If installed correctly, MATLAB shows the installed toolbox path for `unisparse.m`.

### Use directly (no init_project)

Users can now call:

```matlab
results = unisparse(X, y, [1e-3, 1e1], 5, 'all');
```

No manual `addpath`, no `init_project` required.

---

## 5. Updating toolbox versions

When you release updates:

1. Increase toolbox version (e.g. `1.0.1`, `1.1.0`).
2. Rebuild `.mltbx`.
3. Send new file to users.

User update process:

- install new `.mltbx` (MATLAB handles replacement), or
- uninstall old version first, then install new one.

To manage installed add-ons in MATLAB:

- **Home > Add-Ons > Manage Add-Ons**

---

## 6. Uninstall instructions (for users)

In MATLAB:

1. Go to **Home > Add-Ons > Manage Add-Ons**.
2. Find `UniSparse`.
3. Click uninstall.

Or with command line tools for add-ons (version-dependent).

---

## 7. Recommended hardening before selling

Before packaging for customers, verify:

1. Fresh MATLAB session test:
   - install `.mltbx`,
   - do not run `init_project`,
   - run a simple `unisparse` example.
2. Parallel test:
   - run with a pool open and with no pool open.
3. Missing dependency test:
   - confirm all required functions are inside included folders.
4. Version compatibility note:
   - document minimum MATLAB release (for example R2021b+).

---

## 8. Troubleshooting

### Problem: `Undefined function or variable 'unisparse'`

Fix:

1. Check installation in **Manage Add-Ons**.
2. Run `which unisparse -all`.
3. Reinstall toolbox if not found.

### Problem: helper function not found (`split_data`, `RMPSH`, etc.)

Fix:

1. Toolbox likely missing a folder in package configuration.
2. Re-open toolbox project and include missing folder.
3. Rebuild and reinstall.

### Problem: parallel worker path issue

If users see worker path errors, ask them to restart pool:

```matlab
delete(gcp('nocreate'))
parpool('Processes')
```

Because your `unisparse.m` already has worker `addpath` propagation logic, this usually resolves stale worker sessions.

---

## 9. Suggested customer-facing quick start text

You can copy this into your product page:

1. Install `UniSparse.mltbx` by double-clicking it in MATLAB.
2. Run your script and call `unisparse(...)` directly.
3. No setup scripts are required.

---

## 10. Optional: keep init_project for developers only

You can keep `init_project.m` for your own development/testing.

- Developers using the source tree may still find it convenient.
- End users of the toolbox should not need it.

That gives you both:

- easy internal workflow,
- clean buyer experience.
