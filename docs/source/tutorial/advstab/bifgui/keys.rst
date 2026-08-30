.. _advstabbifguikeys:

Default keyboard shortcuts
~~~~~~~~~~~~~~~~~~~~~~~~~~

The GUI grew out of a purely key-driven tool, and the keys are still the fastest way to use it: a
sweep is mostly :kbd:`Space`, :kbd:`+`, :kbd:`-` and :kbd:`b`, with the menus reserved for the things
you do once. Every shortcut is shown next to its menu entry, so they can be learned by using the menus
first.

Note that upper and lower case are two different commands: :kbd:`a` scales the arclength and
:kbd:`Shift+a` switches that off, :kbd:`t` and :kbd:`Shift+t` leave the branch along different modes.

Continuation
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Command
   * - :kbd:`Space`
     - Step - one arclength continuation step
   * - :kbd:`Shift+Space`
     - Multistep - keep stepping until the branch leaves the visible axes
   * - :kbd:`*`
     - Step, but never grow ``ds``
   * - :kbd:`+` / :kbd:`-`
     - Increase / decrease ``ds`` by a factor 1.25
   * - :kbd:`/`
     - Reverse the direction of travel
   * - :kbd:`a` / :kbd:`Shift+a`
     - Scale the arclength / do not scale it
   * - :kbd:`Esc`
     - Abort the running sweep, between steps

Bifurcations
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Command
   * - :kbd:`b`
     - Locate the bifurcation - or, at one, take the way off it that its type offers
   * - :kbd:`p`
     - Locate a pitchfork with the pitchfork tracker
   * - :kbd:`t` / :kbd:`Shift+t`
     - Leave the branch transiently along eigenmode 0 / 1

Points and branches
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Command
   * - :kbd:`PageUp` / :kbd:`PageDown`
     - Select the next / previous point on the branch
   * - :kbd:`Home` / :kbd:`End`
     - Select the first / last point
   * - :kbd:`Enter`
     - Go to the selected point, i.e. load its state
   * - :kbd:`Backspace`, :kbd:`Delete`
     - Delete the selected point
   * - :kbd:`0` ... :kbd:`9`
     - Tag the point with that number (pressing it again removes the tag)
   * - :kbd:`x` / :kbd:`Shift+x`
     - Split the branch here / merge the selected branch into the current one
   * - :kbd:`m`
     - Toggle move-point mode
   * - :kbd:`g`
     - Grab the selected point, so the selection keys move it along its branch

View and files
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Command
   * - :kbd:`y`
     - Cycle the vertical axis through the available observables
   * - :kbd:`i`
     - Toggle the interpolated splines
   * - :kbd:`o`
     - Export the curves as text columns

Rebinding
^^^^^^^^^

*Settings → Keyboard shortcuts...* opens a dialog listing every command with its current key. Binding
a key that is already taken moves it: one key is one command. What you change is written to a
``keymap.json`` under the user configuration directory
(``~/.config/pyoomph/bifurcation_gui/`` on Linux, ``~/Library/Application Support/pyoomph/bifurcation_gui/``
on macOS, ``%APPDATA%\pyoomph\bifurcation_gui\`` on Windows), so it survives a restart and applies to
every problem. Only the differences from the defaults are stored, so a command that gains a default
binding in a later version of pyoomph gets it. The dialog can also reset everything to the defaults,
and *Help → Shortcut reference* lists whatever is currently in force.

Commands with no default key - most of the menu, including the branch switch and everything to do with
orbits, loci and deflation - can be given one here. So can the custom commands a script registers
through :py:attr:`~pyoomph.utils.bifurcation_gui.BifurcationGUI.custom_key_functions`.
