using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ML
{
    class ColumnNameAttribute : Attribute
    {
        ColumnNameAttribute(string str)
        {
            str = "Score";
        }
    }
}
